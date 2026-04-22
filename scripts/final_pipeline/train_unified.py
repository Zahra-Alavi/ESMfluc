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
import matplotlib.pyplot as plt
from collections import Counter

try:
    import psutil
except ImportError:
    psutil = None

from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr

from data_utils import (
    create_classification_func,
    load_and_preprocess_data,
    load_regression_data,
    SequenceClassificationDataset,
    SequenceRegressionDataset,
    collate_fn_sequence,
    compute_sampling_weights
)

from transformers import EsmModel, EsmTokenizer

try:
    from esm.pretrained import ESM3_sm_open_v0
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    # Patch read-only token properties so transformers' __init__ can set them.
    # Required when esm library version conflicts with installed transformers.
    _SPECIAL_TOK_NAMES = (
        'cls_token', 'eos_token', 'mask_token', 'pad_token',
        'unk_token', 'bos_token', 'sep_token',
    )
    def _make_token_setter(private_name):
        def setter(self, value):
            object.__setattr__(self, private_name, value)
        return setter
    for _tok_name in _SPECIAL_TOK_NAMES:
        _getter = None
        for _klass in EsmSequenceTokenizer.__mro__:
            _cls_attr = _klass.__dict__.get(_tok_name)
            if isinstance(_cls_attr, property):
                _getter = _cls_attr.fget
                break
        if _getter is not None:
            # Always patch on EsmSequenceTokenizer itself (pure-Python class);
            # avoids TypeError if a base class in the MRO is a C-extension type.
            setattr(EsmSequenceTokenizer, _tok_name, property(
                _getter,
                _make_token_setter('_' + _tok_name),
            ))
    _mro_has_getattr = any(
        '__getattr__' in klass.__dict__
        for klass in EsmSequenceTokenizer.__mro__
        if klass is not EsmSequenceTokenizer
    )
    if not _mro_has_getattr:
        def _esm3_compat_getattr(self, name):
            try:
                return object.__getattribute__(self, '_' + name)
            except AttributeError:
                pass
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        EsmSequenceTokenizer.__getattr__ = _esm3_compat_getattr

    ESM3_AVAILABLE = True
except (ImportError, AttributeError, Exception) as _esm3_import_err:
    ESM3_AVAILABLE = False

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
    ESMLinearTokenRegressor,
    ESM3Wrapper,
)

from arguments import parse_arguments


def load_esm_model(model_name, device="cuda"):
    """Load ESM model - supports both ESM2 (transformers) and ESM3 (esm library)"""
    if "esm3" in model_name.lower():
        if not ESM3_AVAILABLE:
            raise ImportError("ESM3 models require the 'esm' library. Install with: pip install esm")
        # ESM3 requires device to be passed during initialization
        model = ESM3_sm_open_v0(device)
        return ESM3Wrapper(model), "esm3"
    else:
        # ESM1/ESM2 from transformers
        model = EsmModel.from_pretrained(f"facebook/{model_name}")
        return model, "esm2"

def load_esm_tokenizer(model_name):
    """Load tokenizer - supports both ESM2 and ESM3"""
    if "esm3" in model_name.lower():
        if not ESM3_AVAILABLE:
            raise ImportError("ESM3 models require the 'esm' library. Install with: pip install esm")
        return EsmSequenceTokenizer()
    else:
        return EsmTokenizer.from_pretrained(f"facebook/{model_name}")

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
                
                # Compute loss with masking (handle padding)
                loss_unreduced = criterion(output, y)
                mask = (y != -100.0)
                loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1)
                
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
    tokenizer = load_esm_tokenizer(args.esm_model)
    
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
        classify_neq = create_classification_func(args.num_classes, args.neq_thresholds)
        train_data = load_and_preprocess_data(
            args.train_data_file, classify_neq
        )
        test_data = load_and_preprocess_data(
            args.test_data_file, classify_neq
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
    
    # Print class distribution for classification
    if args.task_type == "classification":
        print(f"\nClass Distribution (Train):")
        all_train_labels_flat = [lab for sublab in train_dataset.labels for lab in sublab]
        class_counts = Counter(all_train_labels_flat)
        for cls in sorted(class_counts.keys()):
            print(f"  Class {cls}: {class_counts[cls]:,} residues ({class_counts[cls]/len(all_train_labels_flat)*100:.1f}%)")
    
    # Create data loaders
    # Use drop_last for DataParallel to avoid issues with uneven batch sizes across GPUs
    drop_last_train = args.data_parallel and torch.cuda.device_count() > 1
    
    if args.task_type == "classification" and args.oversampling:
        # Use weighted sampling for classification
        weights = compute_sampling_weights(
            train_dataset, args.num_classes, args.neq_thresholds,
            args.oversampling_threshold, args.undersampling_threshold,
            args.oversampling_intensity, args.undersampling_intensity
        )
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                   collate_fn=lambda x: collate_fn_sequence(x, tokenizer),
                                   drop_last=drop_last_train)
        print(f"Using oversampling with weighted sampler (drop_last={drop_last_train})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=lambda x: collate_fn_sequence(x, tokenizer),
                                   drop_last=drop_last_train)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    
    # Load embedding model and move to device
    embedding_model, model_type = load_esm_model(args.esm_model, device=args.device)
    print(f"Loaded {model_type} model: {args.esm_model}")
    
    if args.freeze_all_backbone:
        for param in embedding_model.parameters():
            param.requires_grad = False
        embedding_model.eval()  # Set to eval mode when frozen (disables dropout/batchnorm)
        print("ESM embeddings frozen (all parameters)")
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
        embedding_model.train()  # Set to train mode for fine-tuning
        print(f"Freezing ESM layers {args.freeze_layers}")
    else:
        embedding_model.train()  # Set to train mode for fine-tuning
        print("ESM embeddings will be fine-tuned (all parameters)")
    
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
        # Use reduction='none' for all losses to enable manual masking
        if args.regression_loss == "mse":
            criterion = nn.MSELoss(reduction='none')
        elif args.regression_loss == "mae":
            criterion = nn.L1Loss(reduction='none')
        elif args.regression_loss == "huber":
            criterion = nn.HuberLoss(delta=args.huber_delta, reduction='none')
        elif args.regression_loss == "weighted_mse":
            criterion = nn.MSELoss(reduction='none')  # Use same as mse, masking handled below
        else:
            raise ValueError(f"Unknown regression loss: {args.regression_loss}")
    else:
        if args.loss_function == "focal":
            # Compute class weights if requested
            if getattr(args, 'focal_class_weights', False):
                print("Computing class weights for FocalLoss...")
                all_train_labels_flat = [lab for sublab in train_dataset.labels for lab in sublab]
                class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels_flat), y=all_train_labels_flat)
                alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(args.device)
                print(f"Class weights: {class_weights}")
                criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0, ignore_index=-1)
            else:
                print("Using FocalLoss without class weights")
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
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
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
    
    # Start timing
    train_start_time = time.time()
    
    # Get initial memory usage
    if psutil:
        process = psutil.Process(os.getpid())
        initial_cpu_mem = process.memory_info().rss / (1024 ** 3)  # GB
    if args.device.type == 'cuda':
        initial_gpu_mem = torch.cuda.memory_allocated(args.device) / (1024 ** 3)  # GB
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Results will be saved to: {result_dir}\n")

    accum_steps = max(1, getattr(args, 'gradient_accumulation_steps', 1))
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps "
              f"(effective batch size = {args.batch_size * accum_steps})")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y = batch['labels'].to(args.device)

            is_last_batch = (batch_idx + 1 == len(train_loader))
            do_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch

            if scaler:
                amp_dtype = torch.float16 if getattr(args, 'amp_dtype', 'fp16') == 'fp16' else torch.bfloat16
                with autocast(dtype=amp_dtype):
                    if args.task_type == "regression":
                        output, _ = model(input_ids, attention_mask, return_features="pre")
                        if output.dim() == 3 and output.size(-1) == 1:
                            output = output.squeeze(-1)
                        loss_unreduced = criterion(output, y)
                        mask = (y != -100.0)
                        loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1)
                    else:
                        logits, _ = model(input_ids, attention_mask, return_features="pre")
                        logits_flat = logits.reshape(-1, args.num_classes)
                        y_flat = y.reshape(-1)
                        loss = criterion(logits_flat, y_flat)
                loss = loss / accum_steps
                scaler.scale(loss).backward()
                if do_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                if args.task_type == "regression":
                    output, _ = model(input_ids, attention_mask, return_features="pre")
                    if output.dim() == 3 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                    loss_unreduced = criterion(output, y)
                    mask = (y != -100.0)
                    loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1)
                else:
                    logits, _ = model(input_ids, attention_mask, return_features="pre")
                    logits_flat = logits.reshape(-1, args.num_classes)
                    y_flat = y.reshape(-1)
                    loss = criterion(logits_flat, y_flat)
                (loss / accum_steps).backward()
                if do_step:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * accum_steps  # unscale for logging
        
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
        
        # Early stopping (require meaningful improvement, matching original train.py)
        MIN_DELTA = 1e-3
        improved = (best_val_loss - val_loss) > MIN_DELTA
        if improved:
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
    
    # Calculate training time
    train_end_time = time.time()
    training_time_minutes = (train_end_time - train_start_time) / 60
    print(f"\nTotal training time: {training_time_minutes:.2f} minutes")
    
    # Get final memory usage
    if psutil:
        final_cpu_mem = process.memory_info().rss / (1024 ** 3)
        print(f"CPU memory: {initial_cpu_mem:.2f} GB → {final_cpu_mem:.2f} GB (Δ {final_cpu_mem - initial_cpu_mem:.2f} GB)")
    if args.device.type == 'cuda':
        final_gpu_mem = torch.cuda.memory_allocated(args.device) / (1024 ** 3)
        peak_gpu_mem = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
        print(f"GPU memory: {initial_gpu_mem:.2f} GB → {final_gpu_mem:.2f} GB (Peak: {peak_gpu_mem:.2f} GB)")
    
    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path}")
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(best_model_path, map_location=args.device))
        else:
            model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    else:
        print(f"\nWarning: Best model file not found at {best_model_path}. Using current model state.")
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    epochs_list = [h['epoch'] for h in train_history]
    train_losses = [h['train_loss'] for h in train_history]
    val_losses = [h.get('loss', h.get('val_loss')) for h in train_history]
    
    plt.plot(epochs_list, train_losses, label='Training Loss', marker='o')
    if val_losses:
        plt.plot(epochs_list, val_losses, label='Validation Loss', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    loss_curve_path = os.path.join(result_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")
    
    # Final test evaluation
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    test_metrics = evaluate_model(model, test_loader, criterion, args)
    
    # Save args.txt
    args_path = os.path.join(result_dir, 'args.txt')
    with open(args_path, 'w') as f:
        f.write(str(args))
    print(f"Saved arguments to {args_path}")
    
    if args.task_type == "regression":
        print(f"\nTest Set Results:")
        print(f"  MSE:  {test_metrics['mse']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  Pearson r: {test_metrics['pearson_r']:.4f}")
        print(f"  Total residues: {test_metrics['n_residues']}")
        
        # Save comprehensive metrics CSV (matching train.py format)
        metrics_df = pd.DataFrame([{
            'task': 'regression',
            'architecture': args.architecture,
            'esm_model': args.esm_model,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs_ran': len(train_history),
            'best_val_loss': best_val_loss,
            'test_loss': test_metrics['loss'],
            'test_mse': test_metrics['mse'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'pearson_r': test_metrics['pearson_r'],
            'pearson_p': test_metrics.get('pearson_p', 0.0),
            'n_residues': test_metrics['n_residues'],
            'training_time_minutes': training_time_minutes,
            'peak_gpu_mem_gb': peak_gpu_mem if args.device.type == 'cuda' else None,
            'seed': getattr(args, 'seed', None),
            'device': str(args.device)
        }])
        metrics_csv_path = os.path.join(result_dir, 'run_summary.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved run summary to {metrics_csv_path}")
    else:
        print(f"\nTest Set Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Print classification report
        from sklearn.metrics import classification_report as sk_classification_report
        cls_report_dict = test_metrics['report']
        cls_report_str = sk_classification_report([], [], output_dict=False) if not cls_report_dict else \
                         pd.DataFrame(cls_report_dict).transpose().to_string()
        print("\nClassification Report:")
        print(cls_report_str)
        
        # Save classification report (.txt and .tex)
        report_txt_path = os.path.join(result_dir, 'classification_report.txt')
        with open(report_txt_path, 'w') as f:
            f.write(str(cls_report_dict))
        
        report_df = pd.DataFrame(cls_report_dict).transpose()
        latex_table = report_df.to_latex(float_format="%.2f")
        report_tex_path = os.path.join(result_dir, 'classification_report.tex')
        with open(report_tex_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved classification report to {report_txt_path} and {report_tex_path}")
        
        # Save and display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=test_metrics['confusion_matrix'])
        disp.plot(cmap='Blues')
        plt.title('Test Set Confusion Matrix')
        confusion_path = os.path.join(result_dir, 'confusion_matrix.png')
        plt.savefig(confusion_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {confusion_path}")
        
        # Save comprehensive run_summary.csv for classification
        row = {
            'run_dir': result_dir,
            'task': 'classification',
            'batch_size': args.batch_size,
            'embedding_model': args.esm_model,
            'model_architecture': args.architecture,
            'loss_function': args.loss_function,
            'epochs_ran': len(train_history),
            'best_val_loss': best_val_loss,
            'test_accuracy': test_metrics['accuracy'],
            'training_time_minutes': training_time_minutes,
            'peak_gpu_mem_gb': peak_gpu_mem if args.device.type == 'cuda' else None,
            'seed': getattr(args, 'seed', None),
            'device': str(args.device),
            'lr': args.lr,
        }
        
        # Flatten classification report into columns
        for key, val in cls_report_dict.items():
            if key == "accuracy":
                row["accuracy"] = val
                continue
            if isinstance(val, dict):
                key_safe = str(key).replace(" ", "_")
                for subk, subval in val.items():
                    subk_safe = subk.replace("-", "_")
                    row[f"{key_safe}_{subk_safe}"] = subval
        
        # Flatten confusion matrix
        cm = test_metrics['confusion_matrix']
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                row[f"cm_true_{i}_pred_{j}"] = int(cm[i, j])
        
        summary_df = pd.DataFrame([row])
        summary_csv_path = os.path.join(result_dir, 'run_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved run summary to {summary_csv_path}")
    
    
    print(f"\nTraining completed! Results saved to {result_dir}")
    print("="*70)
    
    return model


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train
    model = train_model(args)
