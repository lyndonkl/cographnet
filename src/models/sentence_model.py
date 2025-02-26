import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.sentence.readout import SentenceReadout
from .layers.sentence.graph_prop import SentenceGraphProp
from torch_geometric.nn import GATConv

class SentenceGraphModel(nn.Module):
    """Sentence-level graph neural network with BiGRU and SwiGLU activation."""
    
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
        
        # Initial projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multiple BiGRU layers
        self.bigru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,  # Half size for bidirectional
                bidirectional=True,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Graph attention and convolution layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GATConv(hidden_dim, hidden_dim),
                'gnn': SentenceGraphProp(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        # Readout layer
        self.readout = SentenceReadout(hidden_dim)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: x * sigmoid(beta * x)"""
        return self.w(x) * F.silu(self.v(x))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sentence features [num_sentences, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights from cosine similarity and position bias [num_edges]
            batch: Batch indices [num_sentences] indicating which graph each sentence belongs to
        """
        # Initial projection
        x = self.input_proj(x)
        
        # Process through BiGRU layers
        for bigru in self.bigru_layers:
            x, _ = bigru(x)
            x = self.swiglu(x)
        
        # Process through GNN layers
        for layer in self.layers:
            attended = layer['attention'](x, edge_index, edge_weight)
            x = layer['gnn'](attended, edge_index, edge_weight)
        
        # Readout to get graph-level representation
        x = self.readout(x, batch)
        
        # Final projection
        return self.output_proj(x) 