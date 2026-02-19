import torch.nn.functional as F
import torch

def _masked(preds, targets, masked_value=-100):
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    mask = (targets_flat != masked_value)
    masked_preds = preds_flat[mask]
    masked_targets = targets_flat[mask]
    return masked_preds, masked_targets

def masked_mse_loss(preds, targets, masked_value=-100):
    masked_preds, masked_targets = _masked(preds, targets, masked_value)
    
    # Calculate MSE only on real amino acids
    return F.mse_loss(masked_preds, masked_targets)

def weighted_masked_mse_loss(preds, targets, weight_threshold=3.0, weight_factor=5.0, masked_value=-100):
    masked_preds, masked_targets = _masked(preds, targets, masked_value)
    
    # Give more importance to high Neq values
    weights = torch.ones_like(masked_targets)
    weights[masked_targets > weight_threshold] = weight_factor
    
    # Calculate weighted MSE
    return (weights * (masked_preds - masked_targets)**2).mean()