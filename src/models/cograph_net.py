import torch
import torch.nn as nn
from .layers.word import WordGraphNetwork
from .layers.sentence import SentenceGraphNetwork
from .layers.fusion import FusionLayer

class CoGraphNet(nn.Module):
    def __init__(self, word_in_channels, sent_in_channels, hidden_channels, num_layers, num_classes, dropout_rate=0.3):
        """
        CoGraphNet integrates word-level and sentence-level graph neural networks,
        and fuses their outputs.
        
        Args:
            word_in_channels (int): Input feature dimension for word nodes.
            sent_in_channels (int): Input feature dimension for sentence nodes.
            hidden_channels (int): Hidden feature dimension.
            num_layers (int): Number of propagation layers / GRU layers.
            num_classes (int): Number of output classes.
        """
        super(CoGraphNet, self).__init__()
        self.word_net = WordGraphNetwork(word_in_channels, hidden_channels, 2, num_classes, dropout_rate)
        self.sent_net = SentenceGraphNetwork(sent_in_channels, hidden_channels, 2, num_classes, dropout_rate)
        self.fusion = FusionLayer(dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        # Optional final classification layer after fusion.
        self.final_mlp = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.Dropout(p=dropout_rate)
        )
    
    def forward(self, word_x, word_edge_index, word_batch, word_edge_weight,
                      sent_x, sent_edge_index, sent_batch, sent_edge_weight):
        """
        Args:
            word_x: Word node features [num_word_nodes, word_in_channels]
            word_edge_index: Word graph connectivity [2, num_word_edges]
            word_batch: Batch vector for word nodes [num_word_nodes]
            word_edge_weight: Optional edge weights for word graph [num_word_edges]
            sent_x: Sentence node features [num_sent_nodes, sent_in_channels]
            sent_edge_index: Sentence graph connectivity [2, num_sent_edges]
            sent_batch: Batch vector for sentence nodes [num_sent_nodes]
            sent_edge_weight: Optional edge weights for sentence graph [num_sent_edges]
        """
        # Get word-level graph output (e.g., via global pooling and MLP)
        x_word = self.word_net(word_x, word_edge_index, word_batch, edge_weight=word_edge_weight)
        # Get sentence-level graph output
        x_sen = self.sent_net(sent_x, sent_edge_index, sent_batch, edge_weight=sent_edge_weight)
        # Fuse the outputs using the fusion layer.
        x_fused = self.fusion(x_word, x_sen)
        x_fused = self.dropout(x_fused)
        # Optionally, pass through a final MLP with softmax to produce probabilities.
        out = self.final_mlp(x_fused)
        return out

#############################################
# Example Usage for CoGraphNet
#############################################
if __name__ == '__main__':
    # For demonstration, we simulate data for a batch of 4 graphs.
    # --- Word Graph Data ---
    num_word_nodes = 100
    word_in_channels = 128
    word_x = torch.randn(num_word_nodes, word_in_channels)
    word_edge_index = torch.randint(0, num_word_nodes, (2, 300))
    word_edge_weight = torch.rand(300)
    word_batch = torch.randint(0, 4, (num_word_nodes,))  # 4 graphs

    # --- Sentence Graph Data ---
    num_sent_nodes = 65  # e.g., variable number of sentences across 4 graphs.
    sent_in_channels = 128
    sent_x = torch.randn(num_sent_nodes, sent_in_channels)
    sent_edge_index = torch.randint(0, num_sent_nodes, (2, 200))
    sent_edge_weight = torch.rand(200)
    # Create a batch vector with variable lengths. For example:
    # Graph 0: 15 sentences, Graph 1: 20, Graph 2: 10, Graph 3: 20.
    sent_batch = torch.tensor([0]*15 + [1]*20 + [2]*10 + [3]*20)

    num_classes = 10
    hidden_channels = 128
    num_layers = 3

    model = CoGraphNet(word_in_channels, sent_in_channels, hidden_channels, num_layers, num_classes)
    logits = model(word_x, word_edge_index, word_batch, word_edge_weight,
                   sent_x, sent_edge_index, sent_batch, sent_edge_weight)
    print("Logits shape:", logits.shape)  # Expected shape: [4, num_classes]