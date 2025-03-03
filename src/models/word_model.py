import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.word.attention import WordAttention
from .layers.word.gnn import WordGNN
from .layers.word.readout import WordReadout
from .layers.word.attention import WordAttention
from .layers.word.swiglu import SwiGLU

class WordGraphModel(nn.Module):
    """Word-level graph neural network with GRU, SwiGLU, and Attention."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=0.2)
            for _ in range(num_layers)
        ])

        self.swiglu = SwiGLU(hidden_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': WordAttention(hidden_dim),
                'gnn': WordGNN(hidden_dim)
            }) for _ in range(num_layers)
        ])

        self.readout = WordReadout(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.swiglu(x)

        for layer in self.layers:
            attended = layer['attention'](x, edge_index, edge_weight)
            x = layer['gnn'](attended, edge_index, edge_weight)

        x = self.readout(x, batch)
        return self.output_proj(x)
