import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """SwiGLU Activation as used in CoGraphNet"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x) * torch.clamp(F.silu(self.v(x)), min=-10, max=10)
