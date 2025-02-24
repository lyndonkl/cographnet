from pathlib import Path
import json
from typing import List, Optional, Tuple, Set
import torch
from torch_geometric.data import Dataset
from .graph_builder import GraphBuilder
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

class DocumentGraphDataset(Dataset):
    """Dataset for loading processed documents and converting them to graph representations."""
    
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
                doc = json.load(f)
                self.documents.append(doc)
                self.categories.add(doc['category'])
        
        # Use provided category mapping or create new one
        self.category_to_idx = category_to_idx if category_to_idx is not None else {
            cat: idx for idx, cat in enumerate(sorted(self.categories))
        }
        
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        """List of files in the raw directory."""
        return [f"doc_{idx}.json" for idx in range(len(self.documents))]

    @property
    def processed_file_names(self) -> List[str]:
        """List of files in the processed directory."""
        return [f'data_{idx}.pt' for idx in range(len(self.documents))]

    def process(self):
        """Process raw documents into graphs and save them."""
        for idx, doc in enumerate(self.documents):
            # Create graph from document text
            data = self.graph_builder.build_graph(doc['text'])
            
            # Convert category to one-hot encoded label
            label_idx = self.category_to_idx[doc['category']]
            data.y = F.one_hot(torch.tensor(label_idx), num_classes=len(self.category_to_idx)).float()
            
            # Apply pre_filter if it exists
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            # Apply pre_transform if it exists
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save processed data
            torch.save(data, self.processed_paths[idx])

    def len(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.processed_file_names)

    def get(self, idx: int):
        """Get a single example from the dataset."""
        data = torch.load(self.processed_paths[idx])
        return data
    
    @property
    def num_classes(self) -> int:
        """Get the number of unique categories."""
        return len(self.category_to_idx)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        class_counts = torch.zeros(self.num_classes)
        for doc in self.documents:
            class_counts[self.category_to_idx[doc['category']]] += 1
            
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        return weights / weights.sum()

def get_all_categories(train_dir: str, val_dir: str, test_dir: str) -> Set[str]:
    """Get all unique categories across all datasets."""
    categories = set()
    for data_dir in [train_dir, val_dir, test_dir]:
        for file in Path(data_dir).glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                categories.add(doc['category'])
    return categories

def create_dataloaders(
    root: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create DataLoader instances for train, validation and test sets.
    
    Args:
        root: Root directory where processed graphs will be saved
        train_dir: Directory containing training documents
        val_dir: Directory containing validation documents
        test_dir: Directory containing test documents
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments passed to DocumentGraphDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
    """
    # First, get all unique categories across all datasets
    all_categories = get_all_categories(train_dir, val_dir, test_dir)
    
    # Create consistent category to index mapping
    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
    num_classes = len(category_to_idx)
    
    # Create datasets with shared category mapping
    train_dataset = DocumentGraphDataset(
        f"{root}/train", 
        train_dir, 
        category_to_idx=category_to_idx,
        **dataset_kwargs
    )
    val_dataset = DocumentGraphDataset(
        f"{root}/val", 
        val_dir, 
        category_to_idx=category_to_idx,
        **dataset_kwargs
    )
    test_dataset = DocumentGraphDataset(
        f"{root}/test", 
        test_dir, 
        category_to_idx=category_to_idx,
        **dataset_kwargs
    )
    
    # Create dataloaders with DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, num_classes 