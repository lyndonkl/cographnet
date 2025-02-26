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
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Word features [num_words, hidden_dim]
            batch: Batch indices [num_words] indicating which graph each word belongs to
        Returns:
            Pooled features [batch_size, hidden_dim*2]
        """
        # First transform embeddings
        x = self.embedding_transform(x)
        
        # Then apply feature attention
        attention_scores = self.feature_attention(x)
        attended_features = x * attention_scores
        
        # Get number of graphs in batch
        batch_size = batch.max().item() + 1
        
        # Initialize output tensors
        max_pool = torch.zeros(batch_size, x.size(1), device=x.device)
        mean_pool = torch.zeros(batch_size, x.size(1), device=x.device)
        
        # Pool for each graph in batch
        for i in range(batch_size):
            mask = (batch == i)
            graph_features = attended_features[mask]
            
            if graph_features.size(0) > 0:  # Check if graph has any nodes
                max_pool[i] = torch.max(graph_features, dim=0)[0]
                mean_pool[i] = torch.mean(graph_features, dim=0)
        
        # Concatenate along feature dimension
        return torch.cat([max_pool, mean_pool], dim=1)  # [batch_size, hidden_dim*2] 