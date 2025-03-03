import torch
import torch.nn as nn
import torch.nn.functional as F 

class FeatureFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, word_features: torch.Tensor, sentence_features: torch.Tensor) -> torch.Tensor:
        fused = (self.alpha1 * word_features + self.alpha2 * sentence_features) / 2
        fused = self.w(fused) * torch.clamp(F.silu(self.v(fused)), min=-10, max=10)
        fused = self.dropout(fused)  # Dropout applied before classification
        return fused
