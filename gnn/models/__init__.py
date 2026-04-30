from .advanced_gnn import AdvGNN
from .sage_gnn import GNN
from .simple_gcn import VanillaGNN
from .stripped_sage_gnn import StrippedSageGNN

def model_factory(model_name, hidden_dim, dropout_rate=0.5):
    if model_name == 'simple':
        return VanillaGNN(hidden_dim)
    elif model_name == 'sage':
        return GNN(hidden_dim)
    elif model_name == 'advanced':
        return AdvGNN(hidden_dim, dropout_rate)
    elif model_name == 'stripped':
        return StrippedSageGNN(hidden_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
__all__ = ['model_factory', 'AdvGNN', 'GNN', 'VanillaGNN', 'StrippedSageGNN']