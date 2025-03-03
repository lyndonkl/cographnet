import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttention(nn.Module):
    """Word attention mechanism with positional weighting."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weight = nn.Linear(hidden_dim, 1)  # Learnable weight
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))  # Scaling factor
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        row, col = edge_index

        # Compute attention scores
        attention_scores = torch.sigmoid(self.attention_weight(x))  # [num_nodes, 1]
        attention_scores = attention_scores.squeeze(-1)  # Remove last dimension

        # Apply edge weights and softmax normalization
        energy = attention_scores[row] * attention_scores[col] * edge_weight
        attention_weights = F.softmax(energy, dim=0)

        # Compute weighted sum of node features
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * attention_weights.unsqueeze(-1))

        return out