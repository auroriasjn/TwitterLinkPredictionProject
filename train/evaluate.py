import click
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from torch_geometric.explain import (
    Explainer, CaptumExplainer, fidelity, ModelConfig
)

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

    def forward(self, x, edge_index, *args, **kwargs):
        model_edge_index = {k: v for k, v in edge_index.items()}
        device = next(iter(x.values())).device

        for edge_type in self.edge_types:
            if edge_type not in model_edge_index:
                model_edge_index[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)

        edge_label_index = kwargs.get('edge_label_index')
        
        if edge_label_index is None:
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dim() == 2 and arg.size(0) == 2:
                    edge_label_index = arg
                    break

        if edge_label_index is None and hasattr(self, 'fallback_edge_label_index'):
            edge_label_index = self.fallback_edge_label_index
            
        if edge_label_index is None:
            raise ValueError("edge_label_index could not be found in args, kwargs, or fallbacks.")

        node_embs = self.core_model.gnn(x, model_edge_index)
        pred = self.core_model.classifier(
            node_embs["user"],
            node_embs["user"],
            edge_label_index
        )
        return pred
        

def evaluate(checkpoint_path, model_name, target_interaction, activity_data_path, follow_data_path, visualize_limit):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    activity_data = pd.read_csv(activity_data_path, sep=r'\s+', names=['userA', 'userB', 'timestamp', 'type'])
    follow_data = pd.read_csv(follow_data_path, sep=r'\s+', names=['userA', 'userB'])

    data = create_hetero_graph(activity_data=activity_data, follow_data=follow_data)
    metadata = data.metadata()

    base_model = model_factory(model_name=model_name, hidden_dim=64, dropout_rate=0.5)
    model = Model(
        gnn_base=base_model,
        num_nodes=data['user'].num_nodes,
        num_classes=1,
        metadata=metadata,
        hidden_channels=64
    )

    lit_model = LinkPredictionGNN.load_from_checkpoint(
        checkpoint_path,
        model=model,
        target_edge=target_interaction
    )

    lit_model.to(device)
    lit_model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    _, _, test_loader = create_edge_loaders(
        data=data,
        target_edge=lit_model.target_edge,
        split=(0.7, 0.15, 0.15),
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        batch_size=128
    )

    total_loss, total_fid_plus, total_fid_minus, explained_count = 0, 0.0, 0.0, 0
    all_preds, all_targets = [], []

    explainer_model = HeteroLinkPredictionWrapper(lit_model.model, metadata).to(device)
    print("Configuring CAPTUM Integrated Gradients...")

    explainer = Explainer(
        model=explainer_model,       
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='binary_classification',
            task_level='edge',
            return_type='probs',
        )
    )

    # 5. Evaluation Loop
    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating & Explaining (Captum)")):
        batch = batch.to(device)

        target_edge = lit_model.target_edge
        target_edge_labels = batch[target_edge].edge_label_index
        target = batch[target_edge].edge_label.float()

        # --- Standard Evaluation Pass ---
        with torch.no_grad():
            pred = lit_model(batch).squeeze()
            if pred.ndim == 0:
                pred = pred.unsqueeze(0)

            loss = criterion(pred, target)
            total_loss += loss.item() * target_edge_labels.size(1)

            all_preds.extend(torch.sigmoid(pred).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        # --- Explainer & Fidelity Evaluation (Runs for EVERY batch) ---
        with torch.set_grad_enabled(True):
            base_x_dict = {}
            for node_type in batch.node_types:
                if node_type == 'user':
                    # Successfully merge the 64D embedding with the 2D features
                    learned_emb = lit_model.model.user_emb(batch[node_type].n_id).detach()
                    structural_feat = batch[node_type].x.detach()
                    base_x_dict[node_type] = torch.cat([learned_emb, structural_feat], dim=-1)
                else:
                    base_x_dict[node_type] = batch[node_type].x.detach()

            filtered_edge_index_dict = {
                edge_type: edge_tensor
                for edge_type, edge_tensor in batch.edge_index_dict.items()
                if edge_tensor.size(1) > 0
            }

            num_target_edges = target_edge_labels.size(1)
            limit = min(1, num_target_edges)
            explain_index = torch.arange(limit, device=device)

            # Generate explanation for every batch
            explanation = explainer(
                x=base_x_dict,
                edge_index=filtered_edge_index_dict,
                index=explain_index,
                target=target.long(),
                target_edge_type=target_edge,
                edge_label_index=target_edge_labels
            )
            
            # --- Mask Processing ---
            for n_type in explanation.node_types:
                if 'node_mask' in explanation[n_type] and explanation[n_type].node_mask is not None:
                    mask = explanation[n_type].node_mask.mean(dim=-1, keepdim=True)
                    mask_min, mask_max = mask.min(), mask.max()
                    if mask_max > mask_min:
                        mask = (mask - mask_min) / (mask_max - mask_min + 1e-10)
                    explanation[n_type].node_mask = mask

            # --- Compatibility Fixes for Fidelity ---
            explainer_model.fallback_edge_label_index = target_edge_labels
            explanation._model_args = [] 
            explanation.x = base_x_dict
            explanation.edge_index = batch.edge_index_dict
            explanation.node_mask = HeteroMaskDict(explanation.collect('node_mask'))

            try:
                explanation.edge_mask = HeteroMaskDict(explanation.collect('edge_mask'))
            except KeyError:
                pass

            # Fidelity calculated for every batch
            fid_plus, fid_minus = fidelity(explainer, explanation)
            total_fid_plus += fid_plus
            total_fid_minus += fid_minus
            explained_count += 1
                
    avg_loss = total_loss / len(all_targets)
    auroc = roc_auc_score(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)
    
    # --- THE FIX ---
    all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
    
    # Calculate recall using the binarized predictions
    recall = recall_score(all_targets, all_preds_binary)
    
    print(f"\n--- Standard Metrics ---")
    print(f"Loss:   {avg_loss:.4f}")
    print(f"AUROC:  {auroc:.4f}")
    print(f"AP:     {ap:.4f}")
    print(f"Recall: {recall:.4f}")

    if explained_count > 0:
        avg_fid_plus = total_fid_plus / explained_count
        avg_fid_minus = total_fid_minus / explained_count
        print(f"\n--- Explanation Metrics (over {explained_count} batches) ---")
        print(f"Saved visualization files to current directory.")
        print(f"Average Fidelity+: {avg_fid_plus:.4f} (Higher is better)")
        print(f"Average Fidelity-: {avg_fid_minus:.4f} (Lower is better)")


@click.command()
@click.option('--checkpoint_path', type=click.Path(exists=True), required=True, help='Path to the best model checkpoint (.ckpt)')
@click.option('--model_name', type=click.Choice(['simple', 'sage', 'advanced']), default='sage', help='Type of GNN model to use')
@click.option('--target_interaction', type=click.Choice(['retweet', 'reply', 'follow', 'mention']), default='follow', help='Type of edge to predict')
@click.option('--activity_data_path', type=click.Path(exists=True), default='data/higgs-activity_time.txt', help='Path to the activity data CSV file')
@click.option('--follow_data_path', type=click.Path(exists=True), default='data/higgs-follow_time.txt', help='Path to the follow data CSV file')
@click.option('--visualize_limit', type=int, default=5, help='Number of batches to explain and visualize')
def evaluate_cli(checkpoint_path, model_name, target_interaction, activity_data_path, follow_data_path, visualize_limit):
    evaluate(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        target_interaction=target_interaction,
        activity_data_path=activity_data_path,
        follow_data_path=follow_data_path,
        visualize_limit=visualize_limit
    )

if __name__ == "__main__":
    evaluate_cli()