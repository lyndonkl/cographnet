import torch
import torch.nn as nn
from .layers.gru_block import GRUBlock
from .layers.graph_attention import GraphAttentionBlock
from .layers.readout import ReadoutLayer

class WordGraphModel(nn.Module):
    """Word-level graph processing model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        swiglu_hidden_dim: int = None
    ):
        super().__init__()
        
        # GRU with SwiGLU
        self.gru = GRUBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            swiglu_hidden_dim=swiglu_hidden_dim
        )
        
        # Use swiglu output dim for subsequent layers if specified
        conv_dim = swiglu_hidden_dim or hidden_dim
        
        # Graph attention
        self.graph_conv = GraphAttentionBlock(conv_dim, conv_dim)
        
        # Readout
        self.readout = ReadoutLayer(conv_dim)
        
        # Final projection
        self.proj = nn.Linear(conv_dim * 2, output_dim)  # *2 due to concat in readout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # GRU processing
        x = self.gru(x)
        
        # Graph convolution
        x = self.graph_conv(x, edge_index, edge_attr)
        
        # Readout
        x = self.readout(x)
        
        # Project to output dimension
        return self.proj(x) 