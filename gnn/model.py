import torch
import torch.nn as nn
from torch_geometric.nn import to_hetero
from .classifier import Classifier

import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, num_nodes: int, num_classes: int, gnn_base, metadata=None, hidden_channels: int=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_nodes, hidden_channels)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        
        self.gnn = gnn_base
        if metadata is not None:
            self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')

        self.classifier = Classifier()

    def forward(self, batch, target_edge):
        x_dict = {}
        for node_type in batch.node_types:
            if node_type == 'user':
                learned_emb = self.user_emb(batch[node_type].n_id)
                
                # Get the pre-calculated structural features (Log-Degree)
                structural_feat = batch[node_type].x
                x_dict[node_type] = torch.cat([learned_emb, structural_feat], dim=-1)
            else:
                x_dict[node_type] = batch[node_type].x

        node_embs = self.gnn(x_dict, batch.edge_index_dict)

        pred = self.classifier(
            node_embs["user"], 
            node_embs["user"], 
            batch[target_edge].edge_label_index
        )
        return pred