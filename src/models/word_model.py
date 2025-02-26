import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.word.attention import WordAttention
from .layers.word.gnn import WordGNN
from .layers.word.readout import WordReadout
from .layers.word.attention import WordAttention

class WordGraphModel(nn.Module):
    """Word-level graph neural network with GRU and SwiGLU activation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3
    ):
        super().__init__()

        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multiple GRU layers
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Stack of GNN layers for multi-hop interactions
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': WordAttention(hidden_dim),
                'gnn': WordGNN(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        # Readout layer
        self.readout = WordReadout(hidden_dim)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: x * sigmoid(beta * x)"""
        return self.w(x) * F.silu(self.v(x))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Initial projection
        x = self.input_proj(x)
        
        # Process through GRU layers
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.swiglu(x)
        
        # Process through GNN layers
        for layer in self.layers:
            attended = layer['attention'](x, edge_index, edge_weight)
            x = layer['gnn'](attended, edge_index, edge_weight)
        
        # Readout to get graph-level representation
        x = self.readout(x, batch)
        
        # Final projection
        return self.output_proj(x) 