import click
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from gnn import model_factory, Model, LinkPredictionGNN
from utils import create_edge_loaders, create_hetero_graph


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        # The validation bar is initialized but not displayed
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar
    

def train(model_name, target_interaction, activity_data_path, follow_data_path, max_epochs, checkpoint_dir):
    # Get data first
    activity_data = pd.read_csv(activity_data_path, sep='\s+', names=['userA', 'userB', 'timestamp', 'type'])
    follow_data = pd.read_csv(follow_data_path, sep='\s+', names=['userA', 'userB'])

    data = create_hetero_graph(activity_data=activity_data, follow_data=follow_data)
    metadata = data.metadata()

    # Instantiate models
    base_model = model_factory(model_name=model_name, hidden_dim=64, dropout_rate=0.7)
    model = Model(gnn_base=base_model, num_nodes=data['user'].num_nodes, num_classes=1, metadata=metadata, hidden_channels=64)
    lit_model = LinkPredictionGNN(model=model, target_edge=target_interaction, lr=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lit_model = lit_model.to(device)

    train_loader, val_loader, _ = create_edge_loaders(
        data=data,
        target_edge=lit_model.target_edge,
        split=(0.7, 0.15, 0.15),
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        batch_size=128
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_ap',
        mode='max',
        dirpath=checkpoint_dir,
        filename=(f'best-gnn-{model_name}-{target_interaction}' + '{epoch:02d}-{val_ap:.4f}'),
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, LitProgressBar()], # Add it here
        accelerator='auto',
        devices=1,
        enable_progress_bar=True # Must remain True for the custom bar to work
    )

    print("Starting training...")
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Training complete. Best model saved at: {checkpoint_callback.best_model_path}")

    return checkpoint_callback.best_model_path


@click.command()
@click.option('--model_name', type=click.Choice(['simple', 'sage', 'advanced', 'stripped']), default='simple', help='Type of GNN model to use')
@click.option('--target_interaction', type=click.Choice(['retweet', 'reply', 'follow', 'mention']), default='follow', help='Type of edge to predict (e.g., "user-item")')
@click.option('--activity_data_path', type=click.Path(exists=True), default='data/higgs-activity_time.txt', help='Path to the activity data CSV file')
@click.option('--follow_data_path', type=click.Path(exists=True), default='data/higgs-follow_time.txt', help='Path to the follow data CSV file')
@click.option('--max_epochs', type=int, default=10, help='Maximum number of training epochs')
@click.option('--checkpoint_dir', type=click.Path(), default='checkpoints/', help='Directory to save model checkpoints')
def train_cli(model_name, target_interaction, activity_data_path, follow_data_path, max_epochs, checkpoint_dir):
    train(model_name, target_interaction, activity_data_path, follow_data_path, max_epochs, checkpoint_dir)


if __name__ == "__main__":
    train_cli()