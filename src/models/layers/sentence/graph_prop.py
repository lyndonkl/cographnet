import torch
import torch.nn as nn

class SentenceGraphProp(nn.Module):
    """
    Sentence graph propagation layer that uses edge weights (cosine similarity * position bias)
    for information transfer between sentences.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.transform = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Get source and target nodes
        row, col = edge_index
        
        # Weight the messages by edge weights (cosine similarity * position bias)
        messages = x[col] * edge_weight.unsqueeze(-1)
        
        # Aggregate messages at target nodes
        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)
        
        # Transform aggregated messages
        return self.transform(out) 