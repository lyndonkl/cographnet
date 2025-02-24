import torch
import torch.nn as nn
from .swiglu import SwiGLU

class GRUBlock(nn.Module):
    """GRU block with SwiGLU activation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, swiglu_hidden_dim: int = None):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # Allow configurable SwiGLU hidden dimension, default to same as hidden_dim
        swiglu_hidden_dim = swiglu_hidden_dim or hidden_dim
        self.swiglu = SwiGLU(hidden_dim, swiglu_hidden_dim)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        x, _ = self.gru(x, h)
        return self.swiglu(x)

class BiGRUBlock(nn.Module):
    """Bidirectional GRU block with SwiGLU activation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, swiglu_hidden_dim: int = None):
        super().__init__()
        self.bigru = nn.GRU(
            input_dim, 
            hidden_dim, 
            bidirectional=True,
            batch_first=True
        )
        # For bidirectional, the output dimension is doubled
        bidirectional_dim = hidden_dim * 2
        # Allow configurable SwiGLU hidden dimension, default to same as bidirectional_dim
        swiglu_hidden_dim = swiglu_hidden_dim or bidirectional_dim
        self.swiglu = SwiGLU(bidirectional_dim, swiglu_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        x, _ = self.bigru(x, h)
        x = self.swiglu(x)
        return self.dropout(x) 