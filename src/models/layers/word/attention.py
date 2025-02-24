import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttention(nn.Module):
    """Word attention mechanism with positional weighting."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Calculate attention scores: As(xi) = σ(Watt·xi + batt)
        attention_scores = torch.sigmoid(self.attention(x))
        
        # Apply attention scores to node features
        attended_features = x * attention_scores
        
        # Use edge weights (which include positional information) for message passing
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, attended_features[col] * edge_weight.unsqueeze(-1))
        
        return out 