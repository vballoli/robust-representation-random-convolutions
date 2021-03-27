import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['consistency_loss']

def consistency_loss(
    lamb: float,
    y1: torch.Tensor, 
    y2: torch.Tensor, 
    y3: torch.Tensor) -> torch.Tensor:
    """
    Consistency loss: 
    
    .. math::
    
        \lambda * \sum_{i=1}^{3} KLDivergence(y_i | y_k)
        
        y_i = (y_1 + y_2 + y_3) / 3

    Args:
        lamb (float): lambda from the paper
        y1: Output after random label
        y2: Output after random label
        y3: Output after random label
    """
    yk = ((y1 + y2 + y3) / 3.).to(y1.dtype).to(y1.device)

    return lamb * (F.kl_div(y1, yk) + F.kl_div(y3, yk) + F.kl_div(y3, yk))
