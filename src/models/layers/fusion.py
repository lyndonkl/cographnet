import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, init_alpha1: float = 0.5, init_alpha2: float = 0.5):
        """
        Fusion layer to combine word-level and sentence-level representations.
        
        Args:
            init_alpha1 (float): Initial weight for word-level representation.
            init_alpha2 (float): Initial weight for sentence-level representation.
        """
        super(FusionLayer, self).__init__()
        # Learnable scalar parameters for weighting the contributions.
        self.alpha1 = nn.Parameter(torch.tensor(init_alpha1, dtype=torch.float))
        self.alpha2 = nn.Parameter(torch.tensor(init_alpha2, dtype=torch.float))
        
    def forward(self, x_word: torch.Tensor, x_sen: torch.Tensor) -> torch.Tensor:
        """
        Fuse the two representations.
        
        Args:
            x_word (torch.Tensor): Word-level representation, shape [batch_size, feature_dim].
            x_sen (torch.Tensor): Sentence-level representation, shape [batch_size, feature_dim].
            
        Returns:
            torch.Tensor: Fused representation, shape [batch_size, feature_dim].
        """
        # Compute weighted sum and average.
        fused = (self.alpha1 * x_word + self.alpha2 * x_sen) / 2.0
        return fused

# Example usage:
if __name__ == '__main__':
    batch_size = 4
    feature_dim = 128
    # Simulated outputs from word and sentence graph readout layers.
    x_word = torch.randn(batch_size, feature_dim)
    x_sen = torch.randn(batch_size, feature_dim)
    
    fusion_layer = FusionLayer()
    x_fused = fusion_layer(x_word, x_sen)
    print("Fused representation shape:", x_fused.shape)  # Expected: [4, 128]
