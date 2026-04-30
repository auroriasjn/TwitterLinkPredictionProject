import torch.nn as nn
from torch_geometric.nn import SAGEConv

# Actual GNN
class StrippedSageGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = SAGEConv(hidden_dim + 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)

        return x
