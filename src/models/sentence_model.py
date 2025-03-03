import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.sentence.readout import SentenceReadout
from .layers.sentence.graph_prop import SentenceGraphProp
from .layers.sentence.swiglu import SwiGLU

class SentenceGraphModel(nn.Module):
    """Sentence-level graph neural network with Bi-GRU, GGNN, and SwiGLU activation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3
    ):
        super().__init__()

        # Initial projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Multiple BiGRU layers
        self.bigru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,  # Half size for bidirectional
                bidirectional=True,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Graph layers using Gated Graph Neural Networks (GGNN)
        self.layers = nn.ModuleList([
            SentenceGraphProp(hidden_dim) for _ in range(num_layers)
        ])

        # SwiGLU Activation
        self.swiglu = SwiGLU(hidden_dim)

        # Readout layer
        self.readout = SentenceReadout(hidden_dim)

        # Final projection
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sentence features [num_sentences, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights from cosine similarity and position bias [num_edges]
            batch: Batch indices [num_sentences] indicating which graph each sentence belongs to
        """
        # Initial projection and stabilization
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # Process through BiGRU layers
        for bigru in self.bigru_layers:
            x, _ = bigru(x)
            x = self.swiglu(x)

        # Process through GNN layers
        for ggnn in self.layers:
            x = ggnn(x, edge_index, edge_weight)

        # Readout to get document-level representation
        x = self.readout(x, batch)

        # Final projection
        return self.output_proj(x)