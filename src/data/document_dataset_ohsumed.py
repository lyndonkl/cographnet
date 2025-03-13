from pathlib import Path
import json
from typing import List, Optional, Tuple, Set
import torch
from torch_geometric.data import Dataset, HeteroData
from torch.utils.data import Subset
from .graph_builder import GraphBuilder
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from torch.distributed import broadcast_object_list, barrier

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

class OhsumedDocumentGraphDataset(Dataset):
    """Dataset for loading processed OHSUMED documents and converting them to graph representations."""
    
    def __init__(
        self, 
        root: str,
        data_dir: str,
        category_to_idx: Optional[dict] = None,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        pre_filter: Optional[callable] = None
    ):
        """Initialize the document dataset.
        
        Args:
            root: Root directory where the processed graphs will be saved
            data_dir: Directory containing processed document JSON files
            category_to_idx: Optional mapping of categories to indices
            transform: Optional transform to be applied on each data object
            pre_transform: Optional transform to be applied on each data object before saving
            pre_filter: Optional filter to be applied on data objects before saving
        """
        self.data_dir = Path(data_dir)
        self.graph_builder = GraphBuilder()
        
        # Load documents and gather categories
        self.documents = []
        self.categories = set()
        for file in self.data_dir.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    doc = json.load(f)
                    if 'text' in doc and 'category' in doc and doc['text'].strip() and doc['category'].strip():
                        self.documents.append(doc)
                        self.categories.add(doc['category'])
                    else:
                        print(f"Skipping invalid document in {file}: missing text or category")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {file}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
        
        if not self.documents:
            raise ValueError(f"No valid documents found in {data_dir}")
        
        self.category_to_idx = category_to_idx if category_to_idx is not None else {
            cat: idx for idx, cat in enumerate(sorted(self.categories))
        }
        
        # Initialize with empty valid_indices - will be loaded from metadata
        self.valid_indices = []
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self) -> List[str]:
        """List of files in the processed directory."""
        # Check if metadata exists and load valid indices
        metadata_path = Path(self.processed_dir) / 'metadata.pt'
        if metadata_path.exists():
            print("Loading existing valid indices from metadata")
            self.valid_indices = torch.load(metadata_path)
            print(f"Found {len(self.valid_indices)} processed documents")
            # Return only the files we know exist
            return ['metadata.pt'] + [f'data_{idx}.pt' for idx in self.valid_indices]
        
        # If no metadata exists, return empty list to trigger processing
        return []

    def process(self):
        """Process raw documents into graphs and save them."""
        print("Processing documents...")
        self.valid_indices = []
        
        # Get process info
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        # Divide documents among processes
        docs_per_process = len(self.documents) // world_size
        start_idx = rank * docs_per_process
        end_idx = start_idx + docs_per_process if rank < world_size - 1 else len(self.documents)
        
        # Process only assigned documents
        for idx in range(start_idx, end_idx):
            try:
                doc = self.documents[idx]
                data = self.graph_builder.build_graph(doc['text'])
                label_idx = self.category_to_idx[doc['category']]
                
                # Store the label index directly instead of one-hot encoding
                data.y = torch.tensor(label_idx, dtype=torch.long)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # Save using original document index
                file_path = Path(self.processed_dir) / f'data_{idx}.pt'
                torch.save(data, file_path)
                self.valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error processing document {idx}: {str(e)}")
                continue
        
        # Wait for all processes to finish processing their documents
        if world_size > 1:
            torch.distributed.barrier()
            
            # Gather valid indices from all processes
            all_indices = [[] for _ in range(world_size)]
            torch.distributed.all_gather_object(all_indices, self.valid_indices)
            
            # Wait for gather to complete
            torch.distributed.barrier()
            
            # Combine all indices
            self.valid_indices = sorted(sum(all_indices, []))

        # Ensure all processes reach this point before saving metadata
        if world_size > 1:
            torch.distributed.barrier()
        
        # Save metadata (only from main process)
        if rank == 0:
            print(f"Saving metadata with {len(self.valid_indices)} valid documents")
            metadata_path = Path(self.processed_dir) / 'metadata.pt'
            torch.save(self.valid_indices, metadata_path)

        # Final barrier to ensure metadata is saved before any process proceeds
        if world_size > 1:
            torch.distributed.barrier()

    def get(self, idx: int):
        """Get a single example from the dataset."""
        # Get the original document index from valid_indices
        doc_idx = self.valid_indices[idx]
        data = torch.load(Path(self.processed_dir) / f'data_{doc_idx}.pt')
        return data
    
    @property
    def num_classes(self) -> int:
        """Get the number of unique categories."""
        return len(self.category_to_idx)

    def len(self) -> int:
        """Get the number of valid documents."""
        return len(self.valid_indices)

def get_all_categories(train_dir: str, test_dir: str) -> Set[str]:
    """Get all unique categories across all datasets."""
    categories = set()

    for data_dir in [train_dir, test_dir]:
        for file in Path(data_dir).glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if 'text' in doc and 'category' in doc and doc['text'].strip() and doc['category'].strip():
                    categories.add(doc['category'])

    return categories

def create_dataloaders_ohsumed(
    root: str,
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
    val_split: float = 0.2,  # Validation split ratio
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create DataLoader instances for train, validation and test sets.
    
    Args:
        root: Root directory where processed graphs will be saved
        train_dir: Directory containing training documents
        test_dir: Directory containing test documents
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation
        **dataset_kwargs: Additional arguments passed to OhsumedDocumentGraphDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
    """
    # First, get all unique categories across all datasets
    all_categories = get_all_categories(train_dir, test_dir)
    
    # Create consistent category to index mapping
    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
    num_classes = len(category_to_idx)
    
    # Create datasets with shared category mapping
    train_full_dataset = OhsumedDocumentGraphDataset(
        f"{root}/train", 
        train_dir, 
        category_to_idx=category_to_idx,
        **dataset_kwargs
    )
    
    test_dataset = OhsumedDocumentGraphDataset(
        f"{root}/test", 
        test_dir, 
        category_to_idx=category_to_idx,
        **dataset_kwargs
    )

    # Extract labels for stratified split
    labels = [train_full_dataset[i].y.item() for i in range(len(train_full_dataset))]

    # **Only Rank 0 performs Stratified Split**
    if rank == 0:
        train_idx, val_idx = train_test_split(
            range(len(train_full_dataset)),
            test_size=val_split,
            stratify=labels,
            random_state=42
        )
    else:
        train_idx, val_idx = None, None  # Placeholder for other ranks

    # **Broadcast split indices to all ranks**
    train_idx_list = [train_idx] if rank == 0 else [None]
    val_idx_list = [val_idx] if rank == 0 else [None]

    broadcast_object_list(train_idx_list, src=0)
    broadcast_object_list(val_idx_list, src=0)

    # **Barrier to ensure all ranks have received indices before proceeding**
    barrier()

    # Assign received indices to all ranks
    train_idx = train_idx_list[0]
    val_idx = val_idx_list[0]

    # Create Subsets
    train_subset = Subset(train_full_dataset, train_idx)
    val_subset = Subset(train_full_dataset, val_idx)
    
    # Create dataloaders with DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=True, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, num_replicas=world_size, rank=rank)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes 