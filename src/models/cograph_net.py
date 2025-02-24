import torch
import torch.nn as nn
from .word_model import WordGraphModel
from .sentence_model import SentenceGraphModel
from .layers.fusion import FeatureFusion

class CoGraphNet(nn.Module):
    """Complete CoGraphNet model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_classes: int,
        swiglu_hidden_dim: int = None
    ):
        super().__init__()
        
        # Word and sentence models
        self.word_model = WordGraphModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            swiglu_hidden_dim=swiglu_hidden_dim
        )
        self.sentence_model = SentenceGraphModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            swiglu_hidden_dim=swiglu_hidden_dim
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(output_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        # Process word graph
        word_out = self.word_model(
            data['word'].x,
            data['word', 'co_occurs', 'word'].edge_index,
            data['word', 'co_occurs', 'word'].edge_attr
        )
        
        # Process sentence graph
        sentence_out = self.sentence_model(
            data['sentence'].x,
            data['sentence', 'related_to', 'sentence'].edge_index,
            data['sentence', 'related_to', 'sentence'].edge_attr
        )
        
        # Fuse features
        fused = self.fusion(word_out, sentence_out)
        
        # Classify
        return self.classifier(fused) 