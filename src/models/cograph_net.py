import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_model import WordGraphModel
from .sentence_model import SentenceGraphModel
from .layers.fusion import FeatureFusion
from torch_geometric.nn import GatedGraphConv

class CoGraphNet(nn.Module):
    """
    CoGraphNet: A dual-graph neural network for text classification.
    
    Processes text through separate word and sentence graphs, then fuses their features.
    Word graph: Uses GRU with SwiGLU and multi-hop message passing
    Sentence graph: Uses BiGRU with SwiGLU and position-aware graph propagation
    Feature fusion: Learned weighted combination of word and sentence features
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_classes: int,
        num_word_layers: int = 3,
        num_sentence_layers: int = 3
    ):
        super().__init__()
        
        # Word graph model - processes word co-occurrence relationships
        self.word_model = WordGraphModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_word_layers
        )
        
        # Sentence graph model - processes sentence relationships with positional bias
        self.sentence_model = SentenceGraphModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_sentence_layers
        )
        
        # Feature fusion - learns to combine word and sentence representations
        self.fusion = FeatureFusion(hidden_dim=output_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),  # Normalize before classification
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()
    
    def _init_weights(self):
        """Apply weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):  # Initialize Linear layers
                nn.init.xavier_uniform_(module.weight)  # Good for deep networks
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Bias = 0 for stability

            elif isinstance(module, nn.GRU):  # Initialize GRU layers
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            elif isinstance(module, nn.Embedding):  # Initialize Embedding layers if present
                nn.init.xavier_uniform_(module.weight)

            elif isinstance(module, GatedGraphConv):  # Handle Gated Graph Convolution
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)  # Xavier Uniform for stability
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: Dictionary containing:
                - word.x: Word features
                - word.edge_index: Word graph edges
                - word.edge_attr: Word edge weights (co-occurrence based)
                - word.batch: Word batch indices
                - sentence.x: Sentence features
                - sentence.edge_index: Sentence graph edges
                - sentence.edge_attr: Sentence edge weights
                - sentence.batch: Sentence batch indices
        """
        # Extract node features directly
        word_x = data['word'].x
        sentence_x = data['sentence'].x

        # Extract batch indices directly
        word_batch = data['word'].batch
        sentence_batch = data['sentence'].batch

        # Extract edge indices and attributes directly
        word_edge_index = data[('word', 'co_occurs', 'word')].edge_index
        word_edge_attr = data[('word', 'co_occurs', 'word')].edge_attr

        sentence_edge_index = data[('sentence', 'related_to', 'sentence')].edge_index
        sentence_edge_attr = data[('sentence', 'related_to', 'sentence')].edge_attr

        # Process word graph
        word_out = self.word_model(
            word_x,
            word_edge_index,
            word_edge_attr,
            word_batch
        )

        # Process sentence graph
        sentence_out = self.sentence_model(
            sentence_x,
            sentence_edge_index,
            sentence_edge_attr,
            sentence_batch
        )

        # Feature fusion
        fused = self.fusion(word_out, sentence_out)
        fused = F.dropout(fused, p=0.2, training=self.training)  # Dropout before classification
        
        # Final classification
        outputs = self.classifier(fused)
        
        return outputs
