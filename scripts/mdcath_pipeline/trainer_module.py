import torch
import lightning as L
from torch.optim import AdamW
from losses import mse_loss, weighted_mse_loss
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
        m_preds, m_targets = self._masked(preds, batch['labels'])
        loss = self._calculate_loss(m_preds, m_targets)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)
        return loss
    
    def _masked(self, preds, targets):
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        mask = (targets_flat != self.masked_value)
        masked_preds = preds_flat[mask]
        masked_targets = targets_flat[mask]
        return masked_preds, masked_targets

    def validation_step(self, batch, batch_idx):
        preds = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            temperature=batch['temperature']
        )

        m_preds, m_targets = self._masked(preds, batch['labels'])
        loss = self._calculate_loss(m_preds, m_targets)
        if m_targets.numel() > 0:
            m_preds = m_preds.float()
            m_targets = m_targets.float()
            self.val_mae.update(m_preds, m_targets)
            self.val_spearman.update(m_preds, m_targets)
            self.val_pearson.update(m_preds, m_targets)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mae", self.val_mae, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_spearman", self.val_spearman, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pearson", self.val_pearson, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_epoch_start(self):
        # Ensure the ESM backbone is in training mode
        self.model.esm.train()
        
    def _calculate_loss(self, preds, targets):
        if self.loss_type == 'weighted':
            return weighted_mse_loss(preds, targets, self.weight_threshold, self.weight_factor)
        else:
            return mse_loss(preds, targets)