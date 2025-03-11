import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv

class WordGNN(nn.Module):
    """Gated Graph Neural Network for word-level propagation."""
    
    def __init__(self, hidden_dim: int, num_steps: int = 3):
        super().__init__()
        self.ggnn = GatedGraphConv(hidden_dim, num_steps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return self.ggnn(x, edge_index, edge_weight)