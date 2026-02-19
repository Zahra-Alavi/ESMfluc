import torch.nn.functional as F
import torch

def mse_loss(preds, targets):
    return F.mse_loss(preds, targets)

def weighted_mse_loss(preds, targets, weight_threshold=3.0, weight_factor=5.0):
    # Give more importance to high Neq values
    weights = torch.ones_like(targets)
    weights[targets > weight_threshold] = weight_factor
    
    return (weights * (preds - targets)**2).mean()