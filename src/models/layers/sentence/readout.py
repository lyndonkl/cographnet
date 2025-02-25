import torch
import torch.nn as nn

class SentenceReadout(nn.Module):
    """Sentence-specific readout with embedding transformation and attention-based pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Embedding transformation
        self.embedding_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Feature attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform embeddings
        x = self.embedding_transform(x)
        
        # Generate Q, K, V projections
        Q = self.query(x)  # [num_nodes, hidden_dim]
        K = self.key(x)    # [num_nodes, hidden_dim]
        V = self.value(x)  # [num_nodes, hidden_dim]
        
        # Calculate attention scores using scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(energy, dim=-1)
        
        # Apply attention weights to values
        attended_features = torch.matmul(attention_weights, V)
        
        # Pool attended features
        max_pool = torch.max(attended_features, dim=0)[0]
        mean_pool = torch.mean(attended_features, dim=0)
        
        return torch.cat([max_pool, mean_pool]) 