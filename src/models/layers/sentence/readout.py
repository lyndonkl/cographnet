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
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sentence features [num_sentences, hidden_dim]
            batch: Batch indices [num_sentences] indicating which graph each sentence belongs to
        Returns:
            Pooled features [batch_size, hidden_dim*2]
        """
        # Transform embeddings
        x = self.embedding_transform(x)
        
        # Generate Q, K, V projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention scores using scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention_weights = torch.softmax(energy, dim=-1)
        
        # Apply attention weights to values
        attended_features = torch.matmul(attention_weights, V)
        
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