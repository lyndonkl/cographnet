import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class WordGNN(MessagePassing):
    """Word-level graph neural network layer."""
    
    def __init__(self, hidden_dim: int):
        super().__init__(aggr='add')  # Specify aggregation method
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Propagate messages along edges
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Weight messages by edge weights
        return x_j * edge_weight.unsqueeze(-1)
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # Transform aggregated messages
        return self.linear(aggr_out) 