import torch
import lightning as L
from torch.optim import AdamW
from losses import masked_mse_loss, weighted_masked_mse_loss
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef

class EsmFlucTrainer(L.LightningModule):
    def __init__(self, model, lr=1e-5, weight_threshold=3.0, weight_factor=5.0, loss_type='weighted', weight_decay=1e-2, masked_value=-100):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_threshold = weight_threshold
        self.weight_factor = weight_factor
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.masked_value = masked_value
        self.save_hyperparameters(ignore=['model'])
        
        self.val_mae = MeanAbsoluteError()
        self.val_spearman = SpearmanCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        
    def training_step(self, batch, batch_idx):
        # Forward pass
        preds = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            temperature=batch['temperature']
        )
        
        # Calculate loss
        loss = None
        if self.loss_type == 'weighted':
            loss = weighted_masked_mse_loss(preds, batch['labels'], self.weight_threshold, self.weight_factor)
        else:
            loss = masked_mse_loss(preds, batch['labels'])
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            temperature=batch['temperature']
        )
        
        loss = None
        if self.loss_type == 'weighted':
            loss = weighted_masked_mse_loss(preds, batch['labels'], self.weight_threshold, self.weight_factor, self.masked_value)
        else:
            loss = masked_mse_loss(preds, batch['labels'], self.masked_value)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mae", self.val_mae, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_spearman", self.val_spearman, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_pearson", self.val_pearson, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay)