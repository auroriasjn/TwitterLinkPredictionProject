import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.1 

    def forward(self, userA: torch.Tensor, userB: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        featA = userA[edge_label_index[0]]
        featB = userB[edge_label_index[1]]

        featA = F.normalize(featA, p=2, dim=-1)
        featB = F.normalize(featB, p=2, dim=-1)
        
        cosine_sim = (featA * featB).sum(dim=-1)
        logits = cosine_sim / self.temperature
        
        return logits
