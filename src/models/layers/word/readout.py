import torch
import torch.nn as nn

class WordReadout(nn.Module):
    """Word-specific readout with embedding transformation and pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Embedding transformation: emb(xi) = ReLU(W_emb·xi + b_emb)
        self.embedding_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # W_emb·xi + b_emb
            nn.ReLU()
        )
        
        # Feature attention: As(xi) = σ(Watt·xi + batt)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Watt·xi + batt
            nn.Sigmoid()  # σ
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First transform embeddings
        x = self.embedding_transform(x)
        
        # Then apply feature attention
        attention_scores = self.feature_attention(x)
        attended_features = x * attention_scores
        
        # Finally pool the attended features
        max_pool = torch.max(attended_features, dim=0)[0]
        mean_pool = torch.mean(attended_features, dim=0)
        
        return torch.cat([max_pool, mean_pool]) 