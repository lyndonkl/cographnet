import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Handle missing components
        word_x = data['word'].x if 'word' in data else torch.zeros_like(data['sentence'].x)
        sentence_x = data['sentence'].x if 'sentence' in data else torch.zeros_like(data['word'].x)

        # Process word graph
        word_out = self.word_model(
            word_x,
            data.get(('word', 'co_occurs', 'word'), {}).get('edge_index', torch.empty(2, 0, dtype=torch.long)),
            data.get(('word', 'co_occurs', 'word'), {}).get('edge_attr', torch.empty(0, dtype=torch.float)),
            data.get('word', {}).get('batch', torch.zeros(word_x.size(0), dtype=torch.long))
        )

        # Process sentence graph
        sentence_out = self.sentence_model(
            sentence_x,
            data.get(('sentence', 'related_to', 'sentence'), {}).get('edge_index', torch.empty(2, 0, dtype=torch.long)),
            data.get(('sentence', 'related_to', 'sentence'), {}).get('edge_attr', torch.empty(0, dtype=torch.float)),
            data.get('sentence', {}).get('batch', torch.zeros(sentence_x.size(0), dtype=torch.long))
        )

        # Feature fusion
        fused = self.fusion(word_out, sentence_out)
        fused = F.dropout(fused, p=0.2, training=self.training)  # Dropout before classification
        
        # Final classification
        outputs = self.classifier(fused)
        
        return outputs
