import click
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.explain import Explainer, fidelity, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer

from utils import create_edge_loaders, create_hetero_graph
from gnn import model_factory, Model, LinkPredictionGNN

class HeteroMaskDict(dict):
    """A magic dictionary that allows PyG's fidelity metric to do `1.0 - mask_dict` safely."""
    def __rsub__(self, other):
        return HeteroMaskDict({
            k: (other - v if v is not None else None) 
            for k, v in self.items()
        })

class HeteroLinkPredictionWrapper(nn.Module):
    def __init__(self, core_model, metadata):
        super().__init__()
        self.core_model = core_model
        self.metadata = metadata
        self.edge_types = metadata[1]

    def forward(self, x, edge_index, **kwargs):
        model_edge_index = {k: v for k, v in edge_index.items()} 
        device = next(iter(x.values())).device
        
        for edge_type in self.edge_types:
            if edge_type not in model_edge_index:
                model_edge_index[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)

        edge_label_index = kwargs.get('edge_label_index')
        if edge_label_index is None:
            raise ValueError("edge_label_index must be passed to the explainer as a kwarg.")

        node_embs = self.core_model.gnn(x, model_edge_index)
        
        pred = self.core_model.classifier(
            node_embs["user"], 
            node_embs["user"], 
            edge_label_index
        )
        return pred

def gnn_evaluate(checkpoint_path, model_name, target_interaction, activity_data_path, follow_data_path, explain_limit):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data & Create Loaders
    print("Loading data...")
    activity_data = pd.read_csv(activity_data_path, sep='\s+', names=['userA', 'userB', 'timestamp', 'type'])
    follow_data = pd.read_csv(follow_data_path, sep='\s+', names=['userA', 'userB'])

    data = create_hetero_graph(activity_data=activity_data, follow_data=follow_data)
    metadata = data.metadata()

    base_model = model_factory(model_name=model_name, hidden_dim=64, dropout_rate=0.5)
    model = Model(gnn_base=base_model, num_nodes=data['user'].num_nodes, num_classes=1, metadata=metadata, hidden_channels=64)
    
    lit_model = LinkPredictionGNN.load_from_checkpoint(checkpoint_path, model=model, target_edge=target_interaction)
    lit_model.to(device)
    lit_model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    # We only need the test loader now!
    _, _, test_loader = create_edge_loaders(
        data=data, target_edge=lit_model.target_edge, split=(0.7, 0.15, 0.15),
        num_neighbors=[20, 10], neg_sampling_ratio=2.0, batch_size=128
    )

    # 2. Configure GNNExplainer
    explainer_model = HeteroLinkPredictionWrapper(lit_model.model, metadata).to(device)
    
    model_config = ModelConfig(
        mode='binary_classification',  
        task_level='edge',             
        return_type='raw',
    )

    explainer = Explainer(
        model=explainer_model,
        # GNNExplainer optimizes masks over 200 epochs per instance
        algorithm=GNNExplainer(epochs=200, lr=0.01), 
        explanation_type='model',
        node_mask_type='attributes', # We get node masks back!
        edge_mask_type='object',     
        model_config=model_config
    )

    # ==========================================
    # EVALUATE & EXPLAIN (INFERENCE)
    # ==========================================
    print("\n--- Evaluating & Explaining Test Set ---")
    total_loss, total_fid_plus, total_fid_minus, explained_count = 0, 0.0, 0.0, 0
    all_preds, all_targets = [], []

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        batch = batch.to(device)
        target_edge = lit_model.target_edge
        target_edge_labels = batch[target_edge].edge_label_index
        target = batch[target_edge].edge_label.float()

        with torch.no_grad():
            pred = lit_model(batch).squeeze()
            if pred.ndim == 0: pred = pred.unsqueeze(0)
            loss = criterion(pred, target)
            total_loss += loss.item() * target_edge_labels.size(1)
            all_preds.extend(torch.sigmoid(pred).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        # Explainer Inference
        if i < explain_limit: 
            # CRITICAL: Gradients must be enabled for GNNExplainer to optimize the masks
            with torch.set_grad_enabled(True):
                base_x_dict = {}
                for node_type in batch.node_types:
                    if node_type == 'user':
                        learned_emb = lit_model.model.user_emb(batch[node_type].n_id).detach()
                        structural_feat = batch[node_type].x.detach()
                        base_x_dict[node_type] = torch.cat([learned_emb, structural_feat], dim=-1)
                    else:
                        base_x_dict[node_type] = batch[node_type].x.detach()
                
                filtered_edge_index_dict = {
                    k: v for k, v in batch.edge_index_dict.items() if v.size(1) > 0
                }
                
                safe_limit = min(32, target_edge_labels.size(1))
                explain_index = torch.arange(safe_limit, device=device)

                explanation = explainer(
                    x=base_x_dict,
                    edge_index=filtered_edge_index_dict, 
                    index=explain_index,                  
                    target=target.long(),                 
                    target_edge_type=target_edge,         
                    edge_label_index=target_edge_labels   
                )
                
                # --- Mask Processing & Normalization ---
                # 1. Process Nodes
                for n_type in explanation.node_types:
                    if 'node_mask' in explanation[n_type] and explanation[n_type].node_mask is not None:
                        # Drop keepdim=True so it becomes a 1D tensor [N]!
                        mask = explanation[n_type].node_mask.mean(dim=-1) 
                        mask_min, mask_max = mask.min(), mask.max()
                        if mask_max > mask_min:
                            mask = (mask - mask_min) / (mask_max - mask_min + 1e-10)
                        
                        # Filter out noise: only keep nodes > 40% importance
                        mask = torch.where(mask > 0.4, mask, torch.zeros_like(mask))
                        explanation[n_type].node_mask = mask

                # 2. Process Edges (This brings the ghost edges back to life)
                for e_type in explanation.edge_types:
                    if 'edge_mask' in explanation[e_type] and explanation[e_type].edge_mask is not None:
                        mask = explanation[e_type].edge_mask
                        mask_min, mask_max = mask.min(), mask.max()
                        if mask_max > mask_min:
                            mask = (mask - mask_min) / (mask_max - mask_min + 1e-10)
                            
                        # Filter out noise: only keep edges > 40% importance
                        mask = torch.where(mask > 0.4, mask, torch.zeros_like(mask))
                        explanation[e_type].edge_mask = mask

                # --- Conditional Visualization ---
                filename = f"gnn_explainer_subgraph_batch_{i}_{model_name}.png"
                explanation.visualize_graph(filename)
                
                # --- PyG Fidelity Compatibility Hacks ---
                explainer_model.fallback_edge_label_index = target_edge_labels
                
                # 1. THE GOLDILOCKS FIX: Explicitly tell fidelity to pass this kwarg!
                explanation._model_args = ['edge_label_index']
                explanation.edge_label_index = target_edge_labels
                
                # Bypass collect() magic
                explanation.x = base_x_dict
                explanation.edge_index = filtered_edge_index_dict
                
                # Extract Node Masks
                explanation.node_mask = HeteroMaskDict(explanation.collect('node_mask'))
                
                # SAFETY NET: Edge Masks
                try:
                    explanation.edge_mask = HeteroMaskDict(explanation.collect('edge_mask'))
                except KeyError:
                    pass
                
                # --- Conditional Visualization ---
                filename = f"gnn_explainer_subgraph_batch_{i}_{model_name}.png"
                
                # PyG will automatically draw and save the highlighted subgraph!
                explanation.visualize_graph(filename)
                
                # Calculate fidelity
                fid_plus, fid_minus = fidelity(explainer, explanation)
                total_fid_plus += fid_plus
                total_fid_minus += fid_minus
                explained_count += 1


    # --- Metrics Calculation ---
    avg_loss = total_loss / len(all_targets)
    auroc = roc_auc_score(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)

    print(f"\n--- Standard Metrics ---")
    print(f"Loss:  {avg_loss:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AP:    {ap:.4f}")

    if explained_count > 0:
        avg_fid_plus = total_fid_plus / explained_count
        avg_fid_minus = total_fid_minus / explained_count
        print(f"\n--- Explanation Metrics (over {explained_count} batches) ---")
        print(f"Average Fidelity+: {avg_fid_plus:.4f} (Higher is better)")
        print(f"Average Fidelity-: {avg_fid_minus:.4f} (Lower is better)")

@click.command()
@click.option('--checkpoint_path', type=click.Path(exists=True), required=True)
@click.option('--model_name', type=click.Choice(['simple', 'sage', 'advanced']), default='simple')
@click.option('--target_interaction', type=click.Choice(['retweet', 'reply', 'follow', 'mention']), default='follow')
@click.option('--activity_data_path', type=click.Path(exists=True), default='data/higgs-activity_time.txt')
@click.option('--follow_data_path', type=click.Path(exists=True), default='data/higgs-follow_time.txt')
@click.option('--explain_limit', type=int, default=5)
def evaluate_cli(checkpoint_path, model_name, target_interaction, activity_data_path, follow_data_path, explain_limit):
    gnn_evaluate(checkpoint_path, model_name, target_interaction, activity_data_path, follow_data_path, explain_limit)

if __name__ == "__main__":
    evaluate_cli()