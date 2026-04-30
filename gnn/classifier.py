import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Lock the temperature to a standard constant (tau = 0.1).
        # This acts as a mathematical cage: Logits can NEVER exceed +/- 10.
        self.temperature = 0.1 

    def forward(self, userA: torch.Tensor, userB: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # 1. Get the un-normalized outputs from the GNN
        featA = userA[edge_label_index[0]]
        featB = userB[edge_label_index[1]]

        # 2. CRITICAL FIX: Normalize the vectors AFTER the GNN
        featA = F.normalize(featA, p=2, dim=-1)
        featB = F.normalize(featB, p=2, dim=-1)

        # 3. Calculate True Cosine Similarity (Strictly bounded [-1, 1])
        cosine_sim = (featA * featB).sum(dim=-1)
        
        # 4. Scale to stable logits
        logits = cosine_sim / self.temperature
        
        return logits