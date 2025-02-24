import torch
import torch.nn as nn

class SentenceReadout(nn.Module):
    """Sentence-specific readout with embedding transformation and pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Embedding transformation
        self.embedding_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform embeddings
        x = self.embedding_transform(x)
        
        # Apply feature attention
        attention_scores = self.feature_attention(x)
        attended_features = x * attention_scores
        
        # Pool attended features
        max_pool = torch.max(attended_features, dim=0)[0]
        mean_pool = torch.mean(attended_features, dim=0)
        
        return torch.cat([max_pool, mean_pool]) 