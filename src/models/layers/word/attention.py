import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttention(nn.Module):
    """Word attention mechanism with positional weighting."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Separate linear transformations for queries, keys, and values
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Generate Q, K, V projections
        Q = self.query(x)  # [num_nodes, hidden_dim]
        K = self.key(x)    # [num_nodes, hidden_dim]
        V = self.value(x)  # [num_nodes, hidden_dim]
        
        # Calculate attention scores using scaled dot-product attention
        # energy = (Q @ K.transpose(-2, -1)) / self.scale
        row, col = edge_index
        
        # Compute attention scores only for connected nodes
        energy = (Q[row] * K[col]).sum(dim=-1) / self.scale
        
        # Apply edge weights and softmax
        attention_weights = F.softmax(energy * edge_weight, dim=0)
        
        # Compute weighted sum of values
        out = torch.zeros_like(x)
        out.index_add_(0, row, V[col] * attention_weights.unsqueeze(-1))
        
        return out 