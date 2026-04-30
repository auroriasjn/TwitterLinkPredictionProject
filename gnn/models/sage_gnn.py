import torch.nn as nn
from torch_geometric.nn import SAGEConv

# Actual GNN
class GNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(hidden_dim + 2, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index)

        return x
