from pathlib import Path
import json
from typing import List, Optional
import torch
from torch_geometric.data import Dataset
from .graph_builder import GraphBuilder

class DocumentDataset(Dataset):
    def __init__(
        self, 
        root: str,
        json_files: List[Path],
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        pre_filter: Optional[callable] = None
    ):
        """Initialize the document dataset.
        
        Args:
            root: Root directory where the dataset should be saved
            json_files: List of paths to JSON files containing documents
            transform: Optional transform to be applied on each data object
            pre_transform: Optional transform to be applied on each data object before saving
            pre_filter: Optional filter to be applied on data objects before saving
        """
        self.json_files = json_files
        self.graph_builder = GraphBuilder()
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        """List of files in the raw directory."""
        return [str(f.name) for f in self.json_files]

    @property
    def processed_file_names(self) -> List[str]:
        """List of files in the processed directory."""
        return [f'data_{idx}.pt' for idx in range(len(self.json_files))]

    def process(self):
        """Process raw data into graphs and save them."""
        documents = self.graph_builder.load_documents(self.json_files)
        
        for idx, doc in enumerate(documents):
            # Create graph from document
            data = self.graph_builder.build_graph(doc)
            
            # Apply pre_filter if it exists
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            # Apply pre_transform if it exists
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save processed data
            torch.save(data, self._processed_paths[idx])

    def len(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.processed_file_names)

    def get(self, idx: int):
        """Get a single example from the dataset."""
        data = torch.load(self._processed_paths[idx])
        return data 