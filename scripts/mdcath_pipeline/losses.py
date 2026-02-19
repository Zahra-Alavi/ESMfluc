import torch.nn.functional as F
import torch

def masked_mse_loss(preds, targets, masked_value=-100):
    mask = (targets != masked_value)

    masked_preds = preds[mask]
    masked_targets = targets[mask]
    
    # Calculate MSE only on real amino acids
    return F.mse_loss(masked_preds, masked_targets)

def weighted_masked_mse_loss(preds, targets, weight_threshold=3.0, weight_factor=5.0, masked_value=-100):
    mask = (targets != masked_value)
    masked_preds = preds[mask]
    masked_targets = targets[mask]
    
    # Give more importance to high Neq values
    weights = torch.ones_like(masked_targets)
    weights[masked_targets > weight_threshold] = weight_factor
    
    # Calculate weighted MSE
    return (weights * (masked_preds - masked_targets)**2).mean()