import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv

class GraphAttentionBlock(nn.Module):
    """Graph attention block using GatedGraphConv."""
    
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.conv = GatedGraphConv(
            out_channels=out_channels,
            num_layers=num_layers
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        return self.conv(x, edge_index, edge_weight=edge_attr) 