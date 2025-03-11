import torch
import torch.nn as nn
import torch.nn.functional as F 

class FeatureFusion(nn.Module):
    def __init__(self, word_dim: int, sentence_dim: int, fusion_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # Projection layers to align dimensions
        self.word_proj = nn.Linear(word_dim, fusion_dim)
        self.sentence_proj = nn.Linear(sentence_dim, fusion_dim)

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

        self.w = nn.Linear(fusion_dim, fusion_dim)
        self.v = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, word_features: torch.Tensor, sentence_features: torch.Tensor) -> torch.Tensor:
        # Project both word and sentence features to a common space
        word_proj = self.word_proj(word_features)
        sentence_proj = self.sentence_proj(sentence_features)

        # Perform weighted fusion
        fused = (self.alpha1 * word_proj + self.alpha2 * sentence_proj) / 2
        fused = self.w(fused) * torch.clamp(F.silu(self.v(fused)), min=-10, max=10)
        fused = self.dropout(fused)  # Dropout applied before classification
        
        return fused
