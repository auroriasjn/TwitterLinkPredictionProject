import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn

# Vanilla GCN
class VanillaGNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(hidden_dim + 2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        return x