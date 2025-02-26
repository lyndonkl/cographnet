import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.word.attention import WordAttention
from .layers.word.gnn import WordGNN
from .layers.word.readout import WordReadout

class WordGraphModel(nn.Module):
    """Word-level graph processing model with GRU and SwiGLU activation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU layer
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # SwiGLU components
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        # Stack of GNN layers for multi-hop interactions
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': WordAttention(hidden_dim),
                'gnn': WordGNN(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        # Readout for graph-level representation
        self.readout = WordReadout(hidden_dim)
        
        # Final projection
        # *2 because readout concatenates max_pool and mean_pool, each of size hidden_dim
        # Example: If hidden_dim=128, readout returns [max_pool(128) | mean_pool(128)] = 256 features
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: x * sigmoid(beta * x)"""
        return self.w(x) * F.silu(self.v(x))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Initial projection
        x = self.input_proj(x)
        
        # GRU processing with SwiGLU activation
        x, _ = self.gru(x)
        x = self.swiglu(x)
        
        # Process through GNN layers for multi-hop interactions
        for layer in self.layers:
            # Apply attention
            attended = layer['attention'](x, edge_index, edge_weight)
            # Apply GNN
            x = layer['gnn'](attended, edge_index, edge_weight)
        
        # Readout to get graph-level representation
        x = self.readout(x, batch)
        
        # Final projection
        return self.output_proj(x) 