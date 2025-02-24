import torch
import torch.nn as nn

class ReadoutLayer(nn.Module):
    """Readout layer combining attention, sum and max pooling."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention scores
        att_scores = self.attention(x)
        attended = x * att_scores
        
        # Pooling
        sum_pool = torch.sum(attended, dim=1)
        max_pool = torch.max(attended, dim=1)[0]
        
        # Combine
        return torch.cat([sum_pool, max_pool], dim=-1) 