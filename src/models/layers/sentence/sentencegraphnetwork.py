import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

#############################################
# Weighted Sentence Aggregation
#############################################
class WeightedSentenceAggregation(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(WeightedSentenceAggregation, self).__init__(aggr=aggr)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops; if edge_weight is provided, add 1 for each self-loop.
        edge_index, tmp_edge_weight = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, torch.ones(x.size(0), device=x.device)], dim=0)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j, edge_weight):
        out = self.lin(x_j)
        if edge_weight is not None:
            out = out * edge_weight.view(-1, 1)
        return out

#############################################
# Custom GRU Cell with SwiGLU Activation
#############################################
class CustomGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Update gate parameters
        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.U_z = nn.Linear(hidden_dim, hidden_dim)
        # Reset gate parameters
        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        # Candidate hidden state parameters
        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.bias_h = nn.Parameter(torch.zeros(hidden_dim))
        # Additional parameters for SwiGLU activation:
        self.lin_gate = nn.Linear(hidden_dim, hidden_dim)
        self.lin_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, h):
        # x: (batch, input_dim); h: (batch, hidden_dim)
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        candidate_input = self.W_h(x) + self.U_h(r * h) + self.bias_h
        # SwiGLU: candidate = Linear(candidate_input) * sigmoid(Gate(candidate_input))
        gate = torch.sigmoid(self.lin_gate(candidate_input))
        candidate = self.lin_linear(candidate_input) * gate
        h_new = z * candidate + (1 - z) * h
        return h_new

#############################################
# Custom BiGRU using the Custom GRU Cell
#############################################
class CustomBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(CustomBiGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()
        for i in range(num_layers):
            curr_input_dim = input_dim if i == 0 else hidden_dim * 2
            self.forward_cells.append(CustomGRUCell(curr_input_dim, hidden_dim))
            self.backward_cells.append(CustomGRUCell(curr_input_dim, hidden_dim))
    
    def forward(self, x):
        """
        x: Tensor of shape (seq_len, batch, input_dim)
        Returns:
           out: Tensor of shape (seq_len, batch, 2*hidden_dim) 
                (concatenation of forward and backward outputs)
        """
        seq_len, batch, _ = x.shape
        for layer in range(self.num_layers):
            # Initialize hidden states as zeros for both directions.
            h_forward = torch.zeros(batch, self.hidden_dim, device=x.device)
            h_backward = torch.zeros(batch, self.hidden_dim, device=x.device)
            
            forward_outputs = []
            backward_outputs = []
            
            # Forward pass
            for t in range(seq_len):
                h_forward = self.forward_cells[layer](x[t], h_forward)
                forward_outputs.append(h_forward.unsqueeze(0))
            forward_outputs = torch.cat(forward_outputs, dim=0)  # (seq_len, batch, hidden_dim)
            
            # Backward pass
            for t in reversed(range(seq_len)):
                h_backward = self.backward_cells[layer](x[t], h_backward)
                backward_outputs.insert(0, h_backward.unsqueeze(0))
            backward_outputs = torch.cat(backward_outputs, dim=0)  # (seq_len, batch, hidden_dim)
            
            # Concatenate forward and backward outputs at each time step
            x = torch.cat([forward_outputs, backward_outputs], dim=2)  # (seq_len, batch, 2*hidden_dim)
        return x

#############################################
# Sentence Graph Network Implementation (Variable-length Sequences)
#############################################
class SentenceGraphNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes):
        super(SentenceGraphNetwork, self).__init__()
        # First, update sentence node embeddings using a weighted message passing layer.
        self.sentence_agg = WeightedSentenceAggregation(in_channels, in_channels)
        
        # Custom BiGRU with SwiGLU (using our custom GRU cells)
        self.bi_gru = CustomBiGRU(in_channels, hidden_channels, num_layers=num_layers)
        # After BiGRU, the output dimension is 2*hidden_channels.
        
        # Attention layer: maps from (2*hidden_channels) -> 1.
        self.att_gate = nn.Linear(hidden_channels * 2, 1)
        # An embedding transformation after attention.
        self.att_emb = nn.Linear(hidden_channels * 2, hidden_channels)
        self.dropout = nn.Dropout(p=0.3)
        # Final MLP for classification.
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        x: Sentence node feature matrix, shape [num_sentence_nodes, in_channels].
        edge_index: Graph connectivity among sentence nodes, shape [2, num_edges].
        batch: Batch vector mapping each sentence node to its graph, shape [num_sentence_nodes].
        edge_weight: Optional edge weight tensor, shape [num_edges].
        """
        # 1. Update sentence node embeddings via weighted message passing.
            # Iteratively update sentence node embeddings (2 GRU steps)
        for _ in range(2):
            x = self.sentence_agg(x, edge_index, edge_weight=edge_weight)
            x = torch.tanh(x)         # Optional non-linearity as per paper (GraphLayer Tanh)
            x = self.dropout(x)       # Apply dropout after each aggregation
        
        # 2. Process each graph separately because sentence counts vary.
        unique_batches = batch.unique(sorted=True)
        pooled_reprs = []
        for b in unique_batches:
            mask = (batch == b)
            x_b = x[mask]  # Sentence nodes for graph b; shape: [L_b, in_channels]
            if x_b.size(0) == 0:
                continue  # Skip empty graphs
            # Assume the sentences are already in correct order for graph b.
            # Reshape to (seq_len, batch=1, in_channels)
            x_b_seq = x_b.unsqueeze(1)
            # 3. Pass through the custom BiGRU.
            x_bi = self.bi_gru(x_b_seq)  # (L_b, 1, 2*hidden_channels)
            x_bi = x_bi.squeeze(1)       # (L_b, 2*hidden_channels)
            # 4. Compute attention scores.
            att_scores = torch.sigmoid(self.att_gate(x_bi))  # (L_b, 1)
            x_att = x_bi * att_scores  # (L_b, 2*hidden_channels)
            x_att = F.relu(self.att_emb(x_att))  # (L_b, hidden_channels)
            x_att = self.dropout(x_att)  # Add dropout before pooling
            # 5. Pooling over variable-length sequence.
            # Global max pooling:
            p_max, _ = x_att.max(dim=0)  # (hidden_channels,)
            # Global mean pooling:
            p_mean = x_att.mean(dim=0)   # (hidden_channels,)
            # Combine pooling results.
            pooled = torch.cat([p_max, p_mean], dim=0)  # (2*hidden_channels,)
            pooled_reprs.append(pooled)
        # Stack pooled representations from all graphs.
        x_pool = torch.stack(pooled_reprs, dim=0)  # (num_graphs, 2*hidden_channels)
        # 6. Final MLP for prediction.
        out = self.mlp(x_pool)  # (num_graphs, num_classes)
        return out

#############################################
# Example Usage for SentenceGraphNetwork with Variable-length Sequences
#############################################
if __name__ == '__main__':
    # Example settings:
    # Let's say we have 4 graphs with variable numbers of sentences.
    # For simplicity, suppose total sentence nodes = 65 (not a multiple of 4).
    num_sentence_nodes = 65
    in_channels = 128
    hidden_channels = 128
    num_layers = 2
    num_classes = 5

    # Random sentence node features.
    x = torch.randn(num_sentence_nodes, in_channels)
    # Dummy edge index for the sentence graph.
    edge_index = torch.randint(0, num_sentence_nodes, (2, 200))
    # Random edge weights.
    edge_weight = torch.rand(200)
    # Create a batch vector with variable lengths.
    # For example, graph 0 has 15 sentences, graph 1 has 20, graph 2 has 10, graph 3 has 20.
    batch = []
    batch += [0] * 15
    batch += [1] * 20
    batch += [2] * 10
    batch += [3] * 20
    batch = torch.tensor(batch)

    model = SentenceGraphNetwork(in_channels, hidden_channels, num_layers, num_classes)
    logits = model(x, edge_index, batch, edge_weight=edge_weight)
    print("Logits shape:", logits.shape)  # Expected shape: [number_of_graphs, num_classes] (i.e. [4, 5])
