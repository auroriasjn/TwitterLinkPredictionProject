import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

def _create_mask(all_edges, all_times, mask):
    """Helper to extract positive edges and times based on a boolean mask."""
    edge_label_index = all_edges[:, mask]
    edge_label_time = all_times[mask]
    return edge_label_index, edge_label_time

def _filter_graph_by_time(base_data: HeteroData, max_time: int):
    """Temporally masks ALL relations so no future edges can be seen during message passing."""
    new_data = base_data.clone()
    for edge_type in new_data.edge_types:
        if hasattr(new_data[edge_type], 'time'):
            times = new_data[edge_type].time
            mask = times <= max_time
            new_data[edge_type].edge_index = new_data[edge_type].edge_index[:, mask]
            new_data[edge_type].time = times[mask]
    return new_data

def create_label_split(data: HeteroData, target_edge: tuple = ('user', 'RT', 'user'), split: tuple = (0.7, 0.15, 0.15)):
    """Splits the target edge chronologically into train, val, and test sets."""
    all_edges = data[target_edge].edge_index
    all_times = data[target_edge].time

    _, sorted_indices = torch.sort(all_times)
    num_edges = len(sorted_indices)

    train_end = int(num_edges * split[0])
    val_end   = int(num_edges * (split[0] + split[1]))

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask   = torch.zeros(num_edges, dtype=torch.bool)
    test_mask  = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[sorted_indices[:train_end]] = True
    val_mask[sorted_indices[train_end:val_end]] = True
    test_mask[sorted_indices[val_end:]] = True

    return {
        'train': _create_mask(all_edges, all_times, train_mask),
        'val': _create_mask(all_edges, all_times, val_mask),
        'test': _create_mask(all_edges, all_times, test_mask)
    }

def create_edge_loaders(
        data: HeteroData, 
        target_edge: tuple = ('user', 'RT', 'user'), 
        split: tuple = (0.7, 0.15, 0.15),
        num_neighbors: list = [20, 10],
        batch_size: int = 1024, # Bumped batch size for embedding stability
        n_workers: int = 4,
        neg_sampling_ratio: float = 2.0 # Standard 1:1 negative sampling
):
    """
    Creates temporal, leak-free DataLoaders for heterogeneous link prediction.
    Uses native PyG uniform negative sampling.
    """
    # 1. Calculate temporal cutoffs based on the target edge distribution
    all_times = data[target_edge].time
    sorted_times, _ = torch.sort(all_times)
    num_edges = len(sorted_times)
    
    train_end_idx = int(num_edges * split[0]) - 1
    val_end_idx = int(num_edges * (split[0] + split[1])) - 1
    
    train_max_time = sorted_times[train_end_idx].item()
    val_max_time = sorted_times[val_end_idx].item()

    # 2. Get temporal label splits
    splits = create_label_split(data, target_edge, split)
    train_pos_edge, train_pos_time = splits['train']   
    val_pos_edge, val_pos_time = splits['val']
    test_pos_edge, test_pos_time = splits['test']

    # 3. Create isolated message-passing graphs to prevent future-peeking
    train_data = _filter_graph_by_time(data, train_max_time)
    val_data = _filter_graph_by_time(data, train_max_time) # Val graph only knows training history
    test_data = _filter_graph_by_time(data, val_max_time)  # Test graph knows train+val history

    # ---------------------------------------------------------
    # LOADER CONFIGURATIONS
    # ---------------------------------------------------------

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=neg_sampling_ratio, # Native PyG uniform sampling
        edge_label_index=(target_edge, train_pos_edge),
        edge_label_time=train_pos_time - 1, 
        time_attr="time",
        batch_size=batch_size, 
        shuffle=True,
        num_workers=n_workers, 
        persistent_workers=True, 
        filter_per_worker=True,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=neg_sampling_ratio, # Native PyG uniform sampling
        edge_label_index=(target_edge, val_pos_edge),
        edge_label_time=val_pos_time - 1, 
        time_attr="time",
        batch_size=batch_size, 
        shuffle=False,
        num_workers=n_workers, 
        persistent_workers=True, 
        filter_per_worker=True
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=neg_sampling_ratio, # Native PyG uniform sampling
        edge_label_index=(target_edge, test_pos_edge),
        edge_label_time=test_pos_time - 1,
        time_attr="time",
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        persistent_workers=True,
        filter_per_worker=True
    )

    return train_loader, val_loader, test_loader