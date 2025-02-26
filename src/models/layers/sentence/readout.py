import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool

class SentenceReadout(nn.Module):
    """Sentence-level readout using feature transformation and global pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),  # Single scalar attention per node
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sentence features [num_sentences, hidden_dim]
            batch: Batch indices [num_sentences] indicating which graph each sentence belongs to
        Returns:
            Pooled features [batch_size, hidden_dim*2]
        """
        x = self.embedding_transform(x)
        attention_scores = self.feature_attention(x)
        attended_features = x * attention_scores

        # Apply global pooling
        max_pool = global_max_pool(attended_features, batch)
        mean_pool = global_mean_pool(attended_features, batch)

        return torch.cat([max_pool, mean_pool], dim=1)  # [batch_size, hidden_dim*2]
