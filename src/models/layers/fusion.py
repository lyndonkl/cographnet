import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Feature fusion layer that learns to combine word and sentence representations.
    
    Implements equation: Mfusion = (α₁Xword + α₂Xsen)/2
    where α₁ and α₂ are learnable parameters.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Initialize learnable weights
        self.alpha1 = nn.Parameter(torch.ones(1))  # Weight for word features
        self.alpha2 = nn.Parameter(torch.ones(1))  # Weight for sentence features
        
    def forward(self, word_features: torch.Tensor, sentence_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse word and sentence features using learned weights.
        
        Args:
            word_features: Output from word-level graph neural network [batch_size, hidden_dim]
            sentence_features: Output from sentence-level graph neural network [batch_size, hidden_dim]
            
        Returns:
            Fused features [batch_size, hidden_dim]
        """
        # Apply weighted combination and normalize
        fused = (self.alpha1 * word_features + self.alpha2 * sentence_features) / 2
        return fused 