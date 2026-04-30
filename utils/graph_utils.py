import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

def create_hetero_graph(activity_data: pd.DataFrame, follow_data: pd.DataFrame):
    data = HeteroData()

    max_user_id = max(
        activity_data['userA'].max(), activity_data['userB'].max(),
        follow_data['userA'].max(), follow_data['userB'].max()
    )
    num_users = int(max_user_id) + 1
    data['user'].num_nodes = num_users

    fl_edges = torch.tensor(follow_data[['userA', 'userB']].values.T, dtype=torch.long)
    data['user', 'FL', 'user'].edge_index = fl_edges
    data['user', 'FL', 'user'].time = torch.zeros(fl_edges.size(1), dtype=torch.long)

    for edge_type in ['RT', 'MT', 'RE']:
        subset = activity_data[activity_data['type'] == edge_type]
        if not subset.empty:
            edges = torch.tensor(subset[['userA', 'userB']].values.T, dtype=torch.long)
            timestamps = torch.tensor(subset[['timestamp']].values, dtype=torch.long).squeeze()
            data['user', edge_type, 'user'].edge_index = edges
            data['user', edge_type, 'user'].time = timestamps

    all_sources = []
    all_targets = []
    
    # Collect all edges to calculate global degree
    for edge_type in data.edge_types:
        all_sources.append(data[edge_type].edge_index[0])
        all_targets.append(data[edge_type].edge_index[1])
        
    all_sources = torch.cat(all_sources)
    all_targets = torch.cat(all_targets)

    # Calculate out-degree (how active they are) and in-degree (how popular they are)
    out_degree = degree(all_sources, num_nodes=num_users).view(-1, 1)
    in_degree = degree(all_targets, num_nodes=num_users).view(-1, 1)
    out_degree_norm = torch.log1p(out_degree)
    in_degree_norm = torch.log1p(in_degree)

    # Class: concat new features
    data['user'].x = torch.cat([out_degree_norm, in_degree_norm], dim=-1)

    return data