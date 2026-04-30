import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class AdvGNN(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.5, aggr='mean'):
        super().__init__()
        
        # Layer 1
        self.conv1 = SAGEConv(hidden_dim + 2, hidden_dim, aggr=aggr) 
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        # --- First Block ---
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # --- Second Block ---
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x