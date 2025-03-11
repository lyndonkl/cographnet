import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

class SwigluGatedGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers, aggr='add'):
        super(SwigluGatedGraphConv, self).__init__(aggr=aggr)  # "add" aggregation.
        self.num_layers = num_layers
        self.out_channels = out_channels

        # Linear transform for message passing: W_a
        self.lin_a = nn.Linear(in_channels, out_channels, bias=False)

        # Update gate parameters: W_z and U_z + bias
        self.lin_z_msg = nn.Linear(out_channels, out_channels)
        self.lin_z_self = nn.Linear(out_channels, out_channels)
        self.bias_z = nn.Parameter(torch.zeros(out_channels))

        # Reset gate parameters: W_r and U_r + bias
        self.lin_r_msg = nn.Linear(out_channels, out_channels)
        self.lin_r_self = nn.Linear(out_channels, out_channels)
        self.bias_r = nn.Parameter(torch.zeros(out_channels))

        # Candidate hidden state parameters: W_h and U_h + bias
        self.lin_h_msg = nn.Linear(out_channels, out_channels)
        self.lin_h_self = nn.Linear(out_channels, out_channels)
        self.bias_h = nn.Parameter(torch.zeros(out_channels))

        # Additional parameters for SwiGLU candidate activation:
        self.lin_h_gate = nn.Linear(out_channels, out_channels)
        self.lin_h_linear = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Optionally, add self-loops to ensure each node sends a message to itself.
        edge_index, tmp_edge_weight = add_self_loops(edge_index, num_nodes=x.size(0))
        # If no edge weights are provided, assume 1 for self-loop and others.
        if edge_weight is not None:
            # Also add a weight of 1 for the self-loops.
            edge_weight = torch.cat([edge_weight, torch.ones(x.size(0), device=x.device)], dim=0)
        else:
            edge_weight = None

        h = x  # h has shape [num_nodes, out_channels] (assume in_channels==out_channels or adjust accordingly)
        for _ in range(self.num_layers):
            # Message passing: aggregate transformed neighbor features.
            # This computes: m_i = sum_{j in N(i)} lin_a(h_j)*edge_weight (if provided)
            m = self.propagate(edge_index, x=h, edge_weight=edge_weight)  # shape: [num_nodes, out_channels]

            # Compute update gate: z = sigmoid(W_z m + U_z h + b_z)
            z = torch.sigmoid(self.lin_z_msg(m) + self.lin_z_self(h) + self.bias_z)

            # Compute reset gate: r = sigmoid(W_r m + U_r h + b_r)
            r = torch.sigmoid(self.lin_r_msg(m) + self.lin_r_self(h) + self.bias_r)

            # Candidate hidden state using SwiGLU instead of tanh:
            candidate_input = self.lin_h_msg(m) + self.lin_h_self(r * h) + self.bias_h
            h_tilde = self.swiglu(candidate_input)

            # Final update: h = z * h_tilde + (1 - z) * h
            h = z * h_tilde + (1 - z) * h

        return h

    def message(self, x_j, edge_weight):
        # x_j: neighbor node features.
        out = self.lin_a(x_j)
        if edge_weight is not None:
            # edge_weight is assumed to be of shape [num_edges]
            # Reshape to [num_edges, 1] and multiply elementwise.
            out = out * edge_weight.view(-1, 1)
        return out

    def swiglu(self, x):
        """
        Implements the SwiGLU activation:
          SwiGLU(x) = (Linear(x)) * sigmoid(Gate(x))
        Here, we use two separate linear transforms.
        """
        gate = torch.sigmoid(self.lin_h_gate(x))
        linear = self.lin_h_linear(x)
        return linear * gate


class WordGraphNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes):
        super(WordGraphNetwork, self).__init__()
        # Use the custom GGNN layer (which includes GRU-style updates with SwiGLU)
        self.ggnn = SwigluGatedGraphConv(in_channels, hidden_channels, num_layers)

        # Attention layer: computes a scalar attention score per node.
        self.att_gate = nn.Linear(hidden_channels, 1)
        # An embedding transformation (with ReLU) after attention.
        self.att_emb = nn.Linear(hidden_channels, hidden_channels)

        # Final MLP for classification (or for fusion with sentence-level output)
        # We combine global max and mean pooling, so input dimension is 2*hidden_channels.
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        x: Node feature matrix, shape [num_nodes, in_channels]
        edge_index: Graph connectivity, shape [2, num_edges]
        batch: Batch vector mapping each node to its graph, shape [num_nodes]
        edge_weight: Optional edge weight tensor, shape [num_edges]
        """
        # Pass through the GGNN block
        x = self.ggnn(x, edge_index, edge_weight=edge_weight)  # Updated node embeddings: [num_nodes, hidden_channels]

        # Compute attention scores per node.
        att_scores = torch.sigmoid(self.att_gate(x))  # [num_nodes, 1]
        x = x * att_scores  # Scale node embeddings by attention scores.

        # Apply an additional ReLU transformation.
        x = F.relu(self.att_emb(x))  # [num_nodes, hidden_channels]

        # Pooling: compute both global max and mean pooling over nodes for each graph.
        x_max = global_max_pool(x, batch)   # shape: [num_graphs, hidden_channels]
        x_mean = global_mean_pool(x, batch) # shape: [num_graphs, hidden_channels]

        # Concatenate the pooled representations.
        x_pool = torch.cat([x_max, x_mean], dim=1)  # shape: [num_graphs, 2 * hidden_channels]

        # Final MLP for prediction.
        out = self.mlp(x_pool)  # shape: [num_graphs, num_classes]
        return out

# Example usage:
if __name__ == '__main__':
    # Suppose we have:
    # - num_nodes = 100, in_channels = 128, hidden_channels = 128, num_layers (GGNN steps) = 3, num_classes = 10.
    num_nodes = 100
    in_channels = 128
    hidden_channels = 128
    num_layers = 3
    num_classes = 10

    # Random node features, edge index, and batch assignments (for 4 graphs)
    x = torch.randn(num_nodes, in_channels)
    # edge_index: shape [2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, 300))
    # Random edge weights as computed from the positional weighting formula.
    edge_weight = torch.rand(300)
    # Batch: randomly assign nodes to one of 4 graphs.
    batch = torch.randint(0, 4, (num_nodes,))

    model = WordGraphNetwork(in_channels, hidden_channels, num_layers, num_classes)
    logits = model(x, edge_index, batch, edge_weight=edge_weight)
    print("Logits shape:", logits.shape)  # Expected: [4, num_classes]
