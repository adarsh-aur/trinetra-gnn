import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
# Assuming RGCNConv is available from torch_geometric.nn
from torch_geometric.nn import RGCNConv 
from torch_geometric.data import Data, Batch
import numpy as np

class HybridCyberGNN(torch.nn.Module):
    """
    Hybrid GNN combining GAT, GCN, GraphSAGE, and RGCN in parallel
    for enhanced feature extraction and detection capacity.
    
    The architecture runs two layers of each GNN type in parallel, concatenates 
    the resulting node embeddings, pools them, and then classifies the graph.
    """
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=2, num_relations=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        
        # --- 1. GAT Branch (Attention-based) ---
        # L1 output dim: hidden_dim * 4. L2 input dim: hidden_dim * 4. L2 output dim: hidden_dim
        self.gat_conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.3)
        self.gat_conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.3)

        # --- 2. GCN Branch (Spectral/Simple) ---
        self.gcn_conv1 = GCNConv(input_dim, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, hidden_dim)

        # --- 3. GraphSAGE Branch (Inductive/Sampling) ---
        self.sage_conv1 = SAGEConv(input_dim, hidden_dim)
        self.sage_conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # --- 4. RGCN Branch (Relational, using single relation for simplicity) ---
        self.rgcn_conv1 = RGCNConv(input_dim, hidden_dim, num_relations)
        self.rgcn_conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        
        # Classifier input dimension: 
        # (GAT output + GCN output + SAGE output + RGCN output) = 4 * hidden_dim
        classifier_input_dim = 4 * hidden_dim 
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(classifier_input_dim, 64), # Increased size for hybrid features
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(64, output_dim)
        )
    
    def _forward_branch(self, conv1, conv2, x, edge_index, edge_type=None):
        """Helper to run a two-layer branch, passing edge_type only to RGCNConv."""
        # Layer 1
        if isinstance(conv1, RGCNConv):
            x = conv1(x, edge_index, edge_type)
        else:
            x = conv1(x, edge_index)
            
        x = F.elu(x)
        
        # Layer 2
        if isinstance(conv2, RGCNConv):
            x = conv2(x, edge_index, edge_type)
        else:
            x = conv2(x, edge_index)
            
        return F.elu(x)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Prepare edge_type (required for RGCN branch)
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            # Default to single relation type '0' for all edges
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        # --- Run all four branches in parallel ---
        x_gat = self._forward_branch(self.gat_conv1, self.gat_conv2, x, edge_index)
        x_gcn = self._forward_branch(self.gcn_conv1, self.gcn_conv2, x, edge_index)
        x_sage = self._forward_branch(self.sage_conv1, self.sage_conv2, x, edge_index)
        x_rgcn = self._forward_branch(self.rgcn_conv1, self.rgcn_conv2, x, edge_index, edge_type)
        
        # Concatenate results from all branches (Node-level combination)
        x_combined = torch.cat([x_gat, x_gcn, x_sage, x_rgcn], dim=1)
        
        # Graph-level pooling on combined embeddings
        x_pooled = global_mean_pool(x_combined, batch)
        
        # Classification
        x_classified = self.classifier(x_pooled)
        return F.log_softmax(x_classified, dim=1)
    
    def get_node_embeddings(self, data):
        """Returns the concatenated node embeddings for visualization."""
        x, edge_index = data.x, data.edge_index
        
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        # Run all four branches to get the final node embeddings
        x_gat = self._forward_branch(self.gat_conv1, self.gat_conv2, x, edge_index)
        x_gcn = self._forward_branch(self.gcn_conv1, self.gcn_conv2, x, edge_index)
        x_sage = self._forward_branch(self.sage_conv1, self.sage_conv2, x, edge_index)
        x_rgcn = self._forward_branch(self.rgcn_conv1, self.rgcn_conv2, x, edge_index, edge_type)

        # Concatenate and return
        x_combined = torch.cat([x_gat, x_gcn, x_sage, x_rgcn], dim=1)
        
        return x_combined.detach().cpu().numpy()