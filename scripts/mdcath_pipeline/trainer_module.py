import time

import torch
import lightning as L
from torch.optim import AdamW
from losses import mse_loss, weighted_mse_loss
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class EsmFlucTrainer(L.LightningModule):
    def __init__(self, model, lr=1e-5, weight_threshold=3.0, weight_factor=5.0, loss_type='weighted', weight_decay=1e-2, masked_value=-100, model_tag="unknown", eval_temp="unknown", data_scope="test_set", use_log_scaling=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_threshold = weight_threshold
        self.weight_factor = weight_factor
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.masked_value = masked_value
        self.use_log_scaling = use_log_scaling
        self.save_hyperparameters(ignore=['model'])
        
        self.val_mae = MeanAbsoluteError()
        self.val_spearman = SpearmanCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        
        self.model_tag = model_tag
        self.eval_temp = eval_temp
        self.data_scope = data_scope
        
        self.all_test_preds = []
        self.all_test_targets = []

    def training_step(self, batch, batch_idx):
        if 'token_ids' in batch:
            # Path for ESM3
            preds = self.model(
                token_ids=batch['token_ids'],
                temperature=batch['temperature']
            )
        else:
            # Path for ESM2
            preds = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                temperature=batch['temperature']
            )
        
        m_preds, m_targets = self._masked(preds, batch['labels'])
        loss = self._calculate_loss(m_preds, m_targets)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
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
            if self.use_log_scaling:
                m_preds = torch.expm1(m_preds)
                m_targets = torch.expm1(m_targets)
            m_preds = m_preds.float()
            m_targets = m_targets.float()
            self.val_mae.update(m_preds, m_targets)
            self.val_spearman.update(m_preds, m_targets)
            self.val_pearson.update(m_preds, m_targets)
            
            if self.trainer.testing:
                self.all_test_preds.append(m_preds.detach().cpu())
                self.all_test_targets.append(m_targets.detach().cpu())

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
        
    def on_test_epoch_end(self):
        """ This runs ONCE after all test batches are finished """
        if not self.all_test_preds:
            return

        full_preds = torch.cat(self.all_test_preds).numpy()
        full_targets = torch.cat(self.all_test_targets).numpy()

        self._create_residual_plot(full_preds, full_targets)
        self.all_test_preds.clear()
        self.all_test_targets.clear()
        
    def _calculate_loss(self, preds, targets):
        if self.loss_type == 'weighted':
            return weighted_mse_loss(preds, targets, self.weight_threshold, self.weight_factor)
        else:
            return mse_loss(preds, targets)
        
    def _create_residual_plot(self, preds, targets):
        plot_dir = "../../plot/residuals_plot/"
        os.makedirs(plot_dir, exist_ok=True)

        residuals = targets - preds
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Comprehensive Title
        fig.suptitle(f"Model: {self.model_tag} | Temp: {self.eval_temp}K | Scope: {self.data_scope}", fontsize=16)
        
        # Plot 1: Residuals vs Ground Truth
        sns.scatterplot(x=targets, y=residuals, alpha=0.2, ax=axes[0], color='teal')
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_title(f"Residuals (N={len(targets)})")
        axes[0].set_xlabel("Ground Truth $N_{eq}$")
        axes[0].set_ylabel("Residual ($N_{eq}^{Target} - N_{eq}^{Pred}$)")
        
        # Plot 2: Distribution
        sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
        axes[1].set_title("Error Distribution")
        axes[1].legend()
        axes[1].set_xlabel("Residual Error")
        
        # Plot 3: Correlation with Performance Metrics
        sns.regplot(x=targets, y=preds, scatter_kws={'alpha':0.1}, line_kws={'color':'red'}, ax=axes[2])
        # Add metrics to the plot title for quick reference
        s_corr = self.val_spearman.compute().item()
        axes[2].set_title(f"Pred vs Actual (Spearman: {s_corr:.3f})")
        axes[2].set_xlabel("Ground Truth $N_{eq}$")
        axes[2].set_ylabel("Predicted $N_{eq}$")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Filename with scope included
        filename = f"resid_{self.model_tag}_{self.eval_temp}K_{self.data_scope}.png"
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.close()
        print(f"Saved comprehensive plot: {filename}")