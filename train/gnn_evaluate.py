import click
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import LinkNeighborLoader
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
        return pred.view(-1)

def gnn_evaluate(checkpoint_path, model_name, target_interaction, activity_data_path, follow_data_path, explain_limit):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    _, _, test_loader = create_edge_loaders(
        data=data, target_edge=lit_model.target_edge, split=(0.7, 0.15, 0.15),
        num_neighbors=[20, 10], neg_sampling_ratio=2.0, batch_size=128
    )

    print("\n--- Evaluating Test Set ---")
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = batch.to(device)
        target_edge = lit_model.target_edge
        target = batch[target_edge].edge_label.float()

        with torch.no_grad():
            pred = lit_model(batch).squeeze()
            if pred.ndim == 0: pred = pred.unsqueeze(0)
            loss = criterion(pred, target)
            
            total_loss += loss.item() * target.size(0)
            all_preds.extend(torch.sigmoid(pred).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    auroc = roc_auc_score(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)

    print(f"\n--- Standard Metrics ---")
    print(f"Loss:  {avg_loss:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AP:    {ap:.4f}")

    if explain_limit > 0:
        print(f"\n--- Setting up Explainer (Option B) ---")
        
        # Grab the first batch to extract some edges we want to explain
        sample_batch = next(iter(test_loader))
        target_edge_labels = sample_batch[target_edge].edge_label_index
        target_labels = sample_batch[target_edge].edge_label
        
        # Usually, we want to explain Positive edges (where a link actually exists)
        pos_mask = target_labels == 1
        edges_to_explain = target_edge_labels[:, pos_mask]
        
        # Cap to explain_limit
        if edges_to_explain.size(1) > explain_limit:
            edges_to_explain = edges_to_explain[:, :explain_limit]
        elif edges_to_explain.size(1) == 0:
            # Fallback if no positive edges in the first batch for some reason
            edges_to_explain = target_edge_labels[:, :explain_limit]

        explain_loader = LinkNeighborLoader(
            data=data, 
            num_neighbors=[30, 20],
            edge_label_index=(target_edge, edges_to_explain),
            edge_label=torch.ones(edges_to_explain.size(1)), 
            batch_size=1,
            shuffle=False
        )
        
        # Configure Explainer
        explainer_model = HeteroLinkPredictionWrapper(lit_model.model, metadata).to(device)
        model_config = ModelConfig(
            mode='binary_classification',  
            task_level='edge',             
            return_type='raw',
        )
        explainer = Explainer(
            model=explainer_model,
            algorithm=GNNExplainer(epochs=200, lr=0.01), 
            explanation_type='model',
            node_mask_type='object', 
            edge_mask_type='object',     
            model_config=model_config
        )

        total_fid_plus, total_fid_minus, explained_count = 0.0, 0.0, 0

        for i, batch in enumerate(tqdm(explain_loader, desc="Explaining")):
            batch = batch.to(device)
            target_edge_labels = batch[target_edge].edge_label_index
            target = batch[target_edge].edge_label.float().view(-1)

            # I will admit that a good chunk of the code below is Gemini-generated, primarily because
            # the actual internals of getting Pytorch Geometric to work with my (personally) created
            # data modules was a pain.
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
                
                explain_index = torch.arange(target_edge_labels.size(1), device=device)

                explanation = explainer(
                    x=base_x_dict,
                    edge_index=filtered_edge_index_dict, 
                    index=explain_index,                   
                    target_edge_type=target_edge,         
                    edge_label_index=target_edge_labels   
                )
                
                # --- Let PyG handle the kwargs naturally ---
                explainer_model.fallback_edge_label_index = target_edge_labels
                explanation._model_args = ['edge_label_index']
                explanation.edge_label_index = target_edge_labels

                explanation.x = base_x_dict
                explanation.edge_index = filtered_edge_index_dict
                
                try:
                    fid_plus, fid_minus = fidelity(explainer, explanation)
                    total_fid_plus += fid_plus
                    total_fid_minus += fid_minus
                    explained_count += 1
                except Exception as e:
                    print(f"\n[Fidelity Failed] Edge {i}: {e}")
                    
                print(f"\n--- Top Explanations for Target Edge {i} ---")
                
                # A: Text Fallback (Highly reliable for HeteroGraphs)
                for e_type in explanation.edge_types:
                    if 'edge_mask' in explanation[e_type]:
                        mask = explanation[e_type].edge_mask
                        if mask is not None and mask.numel() > 0:
                            # Print top 5 most important edges of this type
                            top_k = min(5, mask.size(0))
                            top_vals, top_idx = torch.topk(mask, top_k)
                            print(f"  Top {top_k} important '{e_type[1]}' edges:")
                            for val, idx in zip(top_vals, top_idx):
                                src = explanation[e_type].edge_index[0, idx].item()
                                dst = explanation[e_type].edge_index[1, idx].item()
                                print(f"    User {src} -> User {dst} (Weight: {val.item():.4f})")

                try:
                    for e_type in explanation.edge_types:
                        if 'edge_mask' in explanation[e_type] and explanation[e_type].edge_mask is not None:
                            mask = explanation[e_type].edge_mask
                            explanation[e_type].edge_mask = torch.where(mask > 0.3, mask, torch.zeros_like(mask))
                        
                    filename = f"gnn_explainer_subgraph_edge_{i}_{model_name}_{target_interaction}.png"
                    explanation.visualize_graph(filename)
                except Exception as e:
                    print(f"  [Visualization Skipped]: {e}")

        if explained_count > 0:
            avg_fid_plus = total_fid_plus / explained_count
            avg_fid_minus = total_fid_minus / explained_count
            print(f"\n--- Explanation Metrics (over {explained_count} edges) ---")
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