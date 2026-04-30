import pytorch_lightning as pl
import torch
from torchmetrics.classification import (
    MulticlassAUROC, BinaryAUROC, 
    MulticlassAccuracy, BinaryAccuracy,
    MulticlassAveragePrecision, BinaryAveragePrecision
)

INTERACTION_DICT = {
    'retweet': 'RT',
    'reply': 'RE',
    'follow': 'FL',
    'mention': 'MT'
}

class LinkPredictionGNN(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, target_edge: str, num_classes: int = 1, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.target_edge = ('user', INTERACTION_DICT.get(target_edge, target_edge), 'user')
        self.lr = lr
        self.num_classes = num_classes
        
        # Loss function setup
        if self.num_classes > 1:
            self.criterion = torch.nn.CrossEntropyLoss()
            # Multiclass metrics
            self.val_auroc = MulticlassAUROC(num_classes=num_classes)
            self.val_acc = MulticlassAccuracy(num_classes=num_classes)
            self.val_ap = MulticlassAveragePrecision(num_classes=num_classes)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
            # Binary metrics
            self.val_auroc = BinaryAUROC()
            self.val_acc = BinaryAccuracy()
            self.val_ap = BinaryAveragePrecision()

    def forward(self, batch):
        return self.model(batch, self.target_edge)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        
        if self.num_classes == 1:
            targets = batch[self.target_edge].edge_label.float()
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.squeeze(1)
            
            # Calculate Binary Accuracy for the batch
            preds_binary = (torch.sigmoid(preds) > 0.5).float()
            acc = (preds_binary == targets).float().mean()
            
        else:
            targets = batch[self.target_edge].edge_label.long()
            
            # Calculate Multiclass Accuracy for the batch
            preds_class = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            acc = (preds_class == targets).float().mean()
        
        loss = self.criterion(preds, targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        
        return loss


    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        
        # Type alignment for loss calculation
        if self.num_classes == 1:
            targets = batch[self.target_edge].edge_label.float()
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.squeeze(1)
        else:
            targets = batch[self.target_edge].edge_label.long()
        
        loss = self.criterion(preds, targets)
        self.log('val_loss', loss, prog_bar=True, batch_size=targets.size(0))
        
        # Calculate Probabilities
        probs = torch.sigmoid(preds) if self.num_classes == 1 else torch.softmax(preds, dim=1)
        metric_targets = targets.long()

        # Update all metrics
        self.val_auroc.update(probs, metric_targets)
        self.val_acc.update(probs, metric_targets)
        self.val_ap.update(probs, metric_targets)
            
        return loss

    def on_validation_epoch_end(self):
        val_auroc = self.val_auroc.compute()
        val_acc = self.val_acc.compute()
        val_ap = self.val_ap.compute()

        self.log('val_auroc', val_auroc, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_ap', val_ap, prog_bar=True)

        # Reset for next epoch
        self.val_auroc.reset()
        self.val_acc.reset()
        self.val_ap.reset()

    def configure_optimizers(self):
        # Separate the model parameters into two groups.
        param_optimizer = list(self.model.named_parameters())
        
        no_decay = ['user_emb.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-4, # Keep decay for GNN and MLP
                'lr': 1e-4            
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,  # 0 decay for embeddings
                'lr': 1e-4           
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }