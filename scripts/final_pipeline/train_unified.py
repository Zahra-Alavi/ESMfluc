#!/usr/bin/env python3
"""
Unified training script for both classification and regression.
Cleaned up version that reduces code duplication.
"""

import os
import datetime
import time
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from data_utils import (
    load_and_preprocess_data,
    load_regression_data,
    SequenceClassificationDataset,
    SequenceRegressionDataset,
    collate_fn_sequence,
    compute_sampling_weights
)

from transformers import EsmModel, EsmTokenizer

from models import (
    FocalLoss,
    WeightedMSELoss,
    BiLSTMClassificationModel,
    BiLSTMWithSelfAttentionModel,
    BiLSTMRegressionModel,
    BiLSTMWithSelfAttentionRegressionModel,
    TransformerClassificationModel,
    TransformerRegressionModel,
    ESMLinearTokenClassifier,
    ESMLinearTokenRegressor
)

from arguments import parse_arguments


def tokenize(sequences, tokenizer):
    """Tokenize sequences without padding"""
    return [tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False) for seq in sequences]


# =============================================================================
# Unified Evaluation Functions
# =============================================================================

def evaluate_model(model, data_loader, criterion, args):
    """
    Unified evaluation for both classification and regression.
    Returns task-specific metrics.
    """
    model.eval()
    m = model.module if isinstance(model, nn.DataParallel) else model
    
    is_regression = args.task_type == "regression"
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y = batch['labels'].to(args.device)
            
            # Forward pass
            if is_regression:
                output, feats = m(input_ids, attention_mask, return_features="pre")
                if output.dim() == 3 and output.size(-1) == 1:
                    output = output.squeeze(-1)
                loss = criterion(output, y)
                
                # Collect predictions (flatten and mask)
                output_flat = output.reshape(-1)
                y_flat = y.reshape(-1)
                mask = y_flat != -100.0
                
                # For log targets, keep everything in log space for metrics
                # (R² in log space is more meaningful - measures fit where model was trained)
                all_preds.extend(output_flat[mask].cpu().numpy())
                all_targets.extend(y_flat[mask].cpu().numpy())
            else:
                # Classification
                logits, feats = m(input_ids, attention_mask, return_features="pre")
                logits_flat = logits.reshape(-1, args.num_classes)
                y_flat = y.reshape(-1)
                loss = criterion(logits_flat, y_flat)
                
                # Collect predictions
                probs = torch.softmax(logits, -1)
                preds = probs.argmax(-1).reshape(-1)
                mask = y_flat != -1
                all_preds.extend(preds[mask].cpu().numpy())
                all_targets.extend(y_flat[mask].cpu().numpy())
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    
    # Compute task-specific metrics
    if is_regression:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_preds)
        pearson_r, _ = pearsonr(all_targets, all_preds)
        
        return {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson_r,
            'n_residues': len(all_preds)
        }
    else:
        # Classification metrics
        report = classification_report(all_targets, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        return {
            'loss': avg_loss,
            'report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': report['accuracy']
        }


# =============================================================================
# Unified Training Loop
# =============================================================================

def train_model(args):
    """
    Unified training function for both classification and regression.
    """
    
    print("\n" + "="*70)
    print(f"TRAINING: {args.task_type.upper()}")
    print(f"Model: {args.architecture}")
    print(f"ESM: {args.esm_model}")
    if args.task_type == "regression":
        print(f"Using log(Neq): {args.use_log_neq}")
        print(f"Activation: {args.activation}")
        print(f"Loss: {args.regression_loss}")
    else:
        print(f"Classes: {args.num_classes}")
        print(f"Loss: {args.loss_function}")
    print("="*70 + "\n")
    
    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    
    # Load tokenizer
    tokenizer = EsmTokenizer.from_pretrained(f"facebook/{args.esm_model}")
    
    # Load data based on task type
    if args.task_type == "regression":
        # Regression: continuous Neq values
        train_data = load_regression_data(args.train_data_file)
        test_data = load_regression_data(args.test_data_file)
        
        # Apply log transform if requested
        if args.use_log_neq:
            print("Transforming Neq to log space...")
            train_data['neq'] = train_data['neq'].apply(lambda x: [np.log(val) for val in x])
            test_data['neq'] = test_data['neq'].apply(lambda x: [np.log(val) for val in x])
            print(f"Log Neq range: [{min([min(n) for n in train_data['neq']]):.4f}, "
                  f"{max([max(n) for n in train_data['neq']]):.4f}]")
        else:
            print(f"Neq range: [{min([min(n) for n in train_data['neq']]):.4f}, "
                  f"{max([max(n) for n in train_data['neq']]):.4f}]")
        
        # Tokenize
        X_train = tokenize(train_data['sequence'], tokenizer)
        y_train = train_data['neq'].tolist()
        X_test = tokenize(test_data['sequence'], tokenizer)
        y_test = test_data['neq'].tolist()
        
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.seed
        )
        
        # Create datasets
        train_dataset = SequenceRegressionDataset(X_train, y_train)
        val_dataset = SequenceRegressionDataset(X_val, y_val)
        test_dataset = SequenceRegressionDataset(X_test, y_test)
        
    else:
        # Classification: discrete classes
        train_data = load_and_preprocess_data(
            args.train_data_file, args.neq_thresholds, args.num_classes
        )
        test_data = load_and_preprocess_data(
            args.test_data_file, args.neq_thresholds, args.num_classes
        )
        
        # Tokenize
        X_train = tokenize(train_data['sequence'], tokenizer)
        y_train = train_data['neq_class'].tolist()
        X_test = tokenize(test_data['sequence'], tokenizer)
        y_test = test_data['neq_class'].tolist()
        
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.seed
        )
        
        # Create datasets
        train_dataset = SequenceClassificationDataset(X_train, y_train)
        val_dataset = SequenceClassificationDataset(X_val, y_val)
        test_dataset = SequenceClassificationDataset(X_test, y_test)
    
    print(f"\nData Split:")
    print(f"  Train: {len(train_dataset)} sequences ({len(train_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} sequences ({len(val_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} sequences")
    
    # Create data loaders
    if args.task_type == "classification" and args.oversampling:
        # Use weighted sampling for classification
        weights = compute_sampling_weights(
            train_dataset, args.num_classes, args.neq_thresholds,
            args.oversampling_threshold, args.undersampling_threshold,
            args.oversampling_intensity, args.undersampling_intensity
        )
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                   collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    
    # Load embedding model
    embedding_model = EsmModel.from_pretrained(f"facebook/{args.esm_model}")
    if args.freeze_all_backbone:
        for param in embedding_model.parameters():
            param.requires_grad = False
        print("ESM embeddings frozen")
    elif args.freeze_layers:
        # Ex: '0-5' means freeze layers 0..5, and unfreeze the rest
        start_layer, end_layer = map(int, args.freeze_layers.split("-"))
        freeze_list = range(start_layer, end_layer+1)
        for name, param in embedding_model.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])
                param.requires_grad = not layer_num in freeze_list
            else:
                param.requires_grad = True
        print(f"Freezing ESM layers {args.freeze_layers}")
    else:
        print("ESM embeddings will be fine-tuned")
    
    # Print trainable ESM parameters
    n_trainable_esm = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    n_total_esm = sum(p.numel() for p in embedding_model.parameters())
    print(f"Trainable ESM params: {n_trainable_esm:,} / {n_total_esm:,}")
    
    # Create model based on architecture and task
    if args.task_type == "regression":
        if args.architecture == "bilstm":
            model = BiLSTMRegressionModel(
                embedding_model, args.hidden_size, args.num_layers,
                num_outputs=args.num_outputs, dropout=args.dropout,
                bidirectional=args.bidirectional, activation=args.activation
            )
        elif args.architecture == "bilstm_attention":
            model = BiLSTMWithSelfAttentionRegressionModel(
                embedding_model, args.hidden_size, args.num_layers,
                num_outputs=args.num_outputs, dropout=args.dropout,
                bidirectional=args.bidirectional, activation=args.activation
            )
        elif args.architecture == "transformer":
            model = TransformerRegressionModel(
                embedding_model, args.transformer_nhead,
                args.transformer_num_encoder_layers, args.transformer_dim_feedforward,
                num_outputs=args.num_outputs, dropout=args.dropout
            )
        elif args.architecture == "esm_linear":
            model = ESMLinearTokenRegressor(
                embedding_model, num_outputs=args.num_outputs
            )
        else:
            raise ValueError(f"Architecture {args.architecture} not supported for regression")
    else:
        # Classification models
        if args.architecture == "bilstm":
            model = BiLSTMClassificationModel(
                embedding_model, args.hidden_size, args.num_layers,
                num_classes=args.num_classes, dropout=args.dropout,
                bidirectional=args.bidirectional
            )
        elif args.architecture == "bilstm_attention":
            model = BiLSTMWithSelfAttentionModel(
                embedding_model, args.hidden_size, args.num_layers,
                num_classes=args.num_classes, dropout=args.dropout,
                bidirectional=args.bidirectional
            )
        elif args.architecture == "transformer":
            model = TransformerClassificationModel(
                embedding_model, args.transformer_nhead,
                args.transformer_num_encoder_layers, args.transformer_dim_feedforward,
                num_classes=args.num_classes, dropout=args.dropout
            )
        elif args.architecture == "esm_linear":
            model = ESMLinearTokenClassifier(embedding_model, num_classes=args.num_classes)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")
    
    model = model.to(args.device)
    
    # Wrap in DataParallel if requested and multiple GPUs available
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using nn.DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Print total trainable parameters
    n_trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {n_trainable_total:,} / {n_total:,}")
    print(f"Task-specific head params: {n_trainable_total - n_trainable_esm:,}")
    
    # Create loss function
    if args.task_type == "regression":
        if args.regression_loss == "mse":
            criterion = nn.MSELoss()
        elif args.regression_loss == "mae":
            criterion = nn.L1Loss()
        elif args.regression_loss == "huber":
            criterion = nn.HuberLoss(delta=args.huber_delta)
        elif args.regression_loss == "weighted_mse":
            criterion = WeightedMSELoss(ignore_value=-100.0)
        else:
            raise ValueError(f"Unknown regression loss: {args.regression_loss}")
    else:
        if args.loss_function == "focal":
            criterion = FocalLoss(alpha=None, gamma=2.0, ignore_index=-1)
        elif args.loss_function == "crossentropy":
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise ValueError(f"Unknown loss function: {args.loss_function}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    # Mixed precision
    scaler = GradScaler() if args.mixed_precision and args.device.type == "cuda" else None
    
    # Results directory
    if args.result_foldername == "timestamp":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"./results/{args.task_type}_{args.architecture}_{timestamp}"
    else:
        result_dir = f"./results/{args.result_foldername}"
    os.makedirs(result_dir, exist_ok=True)
    best_model_path = os.path.join(result_dir, "best_model.pth")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Results will be saved to: {result_dir}\n")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y = batch['labels'].to(args.device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    if args.task_type == "regression":
                        output, _ = model(input_ids, attention_mask, return_features="pre")
                        if output.dim() == 3 and output.size(-1) == 1:
                            output = output.squeeze(-1)
                        loss = criterion(output, y)
                    else:
                        logits, _ = model(input_ids, attention_mask, return_features="pre")
                        logits_flat = logits.reshape(-1, args.num_classes)
                        y_flat = y.reshape(-1)
                        loss = criterion(logits_flat, y_flat)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.task_type == "regression":
                    output, _ = model(input_ids, attention_mask, return_features="pre")
                    if output.dim() == 3 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                    loss = criterion(output, y)
                else:
                    logits, _ = model(input_ids, attention_mask, return_features="pre")
                    logits_flat = logits.reshape(-1, args.num_classes)
                    y_flat = y.reshape(-1)
                    loss = criterion(logits_flat, y_flat)
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, args)
        val_loss = val_metrics['loss']
        
        # Print progress
        if args.task_type == "regression":
            print(f"[Epoch {epoch+1}/{args.epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}, "
                  f"R²: {val_metrics['r2']:.4f}")
        else:
            print(f"[Epoch {epoch+1}/{args.epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            **val_metrics
        })
        
        # Learning rate scheduling
        if scheduler:
            if args.lr_scheduler == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Handle DataParallel model saving
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, best_model_path)
            print(f"  → Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    print(f"\nLoading best model from {best_model_path}")
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(best_model_path, map_location=args.device))
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    
    # Final test evaluation
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    test_metrics = evaluate_model(model, test_loader, criterion, args)
    
    if args.task_type == "regression":
        print(f"Test Set Results:")
        print(f"  MSE:  {test_metrics['mse']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  Pearson r: {test_metrics['pearson_r']:.4f}")
        print(f"  Total residues: {test_metrics['n_residues']}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([{
            'mse': test_metrics['mse'],
            'rmse': test_metrics['rmse'],
            'mae': test_metrics['mae'],
            'r2': test_metrics['r2'],
            'pearson_r': test_metrics['pearson_r'],
            'n_residues': test_metrics['n_residues']
        }])
        metrics_csv_path = os.path.join(result_dir, 'regression_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nSaved metrics to {metrics_csv_path}")
    else:
        print(f"Test Set Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report([], [], output_dict=False))
        
        # Save and display confusion matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        
        disp = ConfusionMatrixDisplay(confusion_matrix=test_metrics['confusion_matrix'])
        disp.plot(cmap='Blues')
        plt.title('Test Set Confusion Matrix')
        confusion_path = os.path.join(result_dir, 'confusion_matrix.png')
        plt.savefig(confusion_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved confusion matrix to {confusion_path}")
        
        # Save classification report
        report_df = pd.DataFrame(test_metrics['report']).transpose()
        report_csv_path = os.path.join(result_dir, 'classification_report.csv')
        report_df.to_csv(report_csv_path)
        print(f"Saved classification report to {report_csv_path}")
    
    # Save results
    # Convert args to JSON-safe dict (exclude device objects)
    args_dict = {k: v for k, v in vars(args).items() if k != 'device'}
    args_dict['device'] = str(args.device)  # Convert device to string
    
    results = {
        'args': args_dict,
        'train_history': train_history,
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(result_dir, 'training_results.json'), 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.device):
                return str(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\nResults saved to {result_dir}")
    print("="*70)
    
    return model, results


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train
    model, results = train_model(args)
