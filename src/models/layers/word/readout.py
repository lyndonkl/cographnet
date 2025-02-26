import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool

class WordReadout(nn.Module):
    """Efficient word-level readout using PyG pooling functions."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Word features [num_words, hidden_dim]
            batch: Batch indices [num_words] indicating which graph each word belongs to
        Returns:
            Pooled features [batch_size, hidden_dim*2]
        """
        # Transform embeddings
        x = self.embedding_transform(x)

        # Apply feature attention
        attention_scores = self.feature_attention(x)
        attended_features = x * attention_scores

        # Apply batch-aware global pooling
        max_pool = global_max_pool(attended_features, batch)
        mean_pool = global_mean_pool(attended_features, batch)

        # Concatenate both pooling outputs
        return torch.cat([max_pool, mean_pool], dim=1)  # [batch_size, hidden_dim*2]
