import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.sentence.readout import SentenceReadout
from .layers.sentence.graph_prop import SentenceGraphProp

class SentenceGraphModel(nn.Module):
    """Sentence-level graph processing model with BiGRU and SwiGLU activation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # BiGRU for sentence processing
        self.bigru = nn.GRU(
            hidden_dim, 
            hidden_dim // 2,  # Half size because bidirectional will double it
            bidirectional=True,
            batch_first=True
        )
        
        # SwiGLU components
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        # Graph propagation layer
        self.graph_prop = SentenceGraphProp(hidden_dim)
        
        # Readout
        self.readout = SentenceReadout(hidden_dim)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        
    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: x * sigmoid(beta * x)"""
        return self.w(x) * F.silu(self.v(x))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sentence features [num_sentences, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights from cosine similarity and position bias [num_edges]
        """
        # Initial projection
        x = self.input_proj(x)
        
        # BiGRU processing
        x, _ = self.bigru(x)
        
        # Apply SwiGLU activation
        x = self.swiglu(x)
        
        # Graph propagation using edge weights
        x = x + self.graph_prop(x, edge_index, edge_weight)  # Residual connection
        
        # Readout to get graph-level representation
        x = self.readout(x)
        
        # Final projection
        return self.output_proj(x) 