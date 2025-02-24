import torch.nn as nn
from .word_model import WordGraphModel
from .sentence_model import SentenceGraphModel
from .layers.fusion import FeatureFusion

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
        num_word_layers: int = 3  # Number of word graph layers
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
            output_dim=output_dim
        )
        
        # Feature fusion - learns to combine word and sentence representations
        self.fusion = FeatureFusion(hidden_dim=output_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: Dictionary containing:
                - word.x: Word features
                - word.edge_index: Word graph edges
                - word.edge_attr: Word edge weights (co-occurrence based)
                - sentence.x: Sentence features
                - sentence.edge_index: Sentence graph edges
                - sentence.edge_weight: Sentence edge weights (cosine similarity * position bias)
        """
        # Process word graph - captures word co-occurrence patterns
        word_out = self.word_model(
            data['word'].x,
            data['word', 'co_occurs', 'word'].edge_index,
            data['word', 'co_occurs', 'word'].edge_attr
        )
        
        # Process sentence graph - captures sentence relationships with position awareness
        sentence_out = self.sentence_model(
            data['sentence'].x,
            data['sentence', 'related_to', 'sentence'].edge_index,
            data['sentence', 'related_to', 'sentence'].edge_weight
        )
        
        # Fuse word and sentence features using learned weights
        fused = self.fusion(word_out, sentence_out)
        
        # Final classification
        return self.classifier(fused) 