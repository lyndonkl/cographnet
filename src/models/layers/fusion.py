import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """Feature fusion layer with learnable parameters."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x_word: torch.Tensor, x_sen: torch.Tensor) -> torch.Tensor:
        return (self.alpha1 * x_word + self.alpha2 * x_sen) / 2 