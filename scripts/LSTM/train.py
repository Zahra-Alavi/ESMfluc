#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:10:41 2025

@author: zalavi
"""

# train.py

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from transformers import EsmModel, EsmTokenizer

from models import (
    FocalLoss, 
    BiLSTMClassificationModel,
    BiLSTMWithSelfAttentionModel
)
from data_utils import load_and_preprocess_data, SequenceClassificationDataset, collate_fn_sequence



# =============================================================================
# Evaluation Utils
# =============================================================================        

def compute_validation_loss(model, data_loader, loss_fn, device, num_classes):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            outputs_flat = outputs.view(-1, num_classes)
            labels_flat = labels.view(-1)

            loss = loss_fn(outputs_flat, labels_flat)
            val_loss += loss.item()
    return val_loss / len(data_loader)

def evaluate_model_and_save_plots(model, data_loader, device, fold, num_classes, run_folder):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=2)
            preds = torch.argmax(probs, dim=2)

            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            mask_flat = (labels_flat != -1)

            valid_preds = preds_flat[mask_flat]
            valid_labels = labels_flat[mask_flat]

            all_preds.extend(valid_preds.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nFold {fold} Classification Report:\n{report}")
    print(f"Fold {fold} Confusion Matrix:\n{cm}")
    
    # Save classification report to a text file
    with open(f'classification_report_fold_{fold}.txt', 'w') as f:
        f.write(f"Fold {fold} Classification Report:\n{report}\n")
        f.write(f"Fold {fold} Confusion Matrix:\n{cm}\n")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold}')
    cm_filename = os.path.join(run_folder, f'confusion_matrix_fold_{fold}.png')
    plt.savefig(cm_filename)
    plt.close()

    return report, cm

# =============================================================================
# Training
# =============================================================================        

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create run folder
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = f"run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    # Save args to text
    with open(os.path.join(run_folder, "run_parameters.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
            
     # Load data
    from data_utils import create_classification_func
    classify_neq = create_classification_func(args.num_classes, args.neq_thresholds)
    training_data = load_and_preprocess_data(args.csv_path, classify_neq)
    
    
# =============================================================================
#     # Sequence-majority label for stratification: to be changed to a better label for each sequence
#     sequence_majority_labels = []
#     for labels in training_data['neq_class']:
#         counts = np.bincount(labels)
#         majority_label = np.argmax(counts)
#         sequence_majority_labels.append(majority_label)
# =============================================================================

    # Tokenizer

    tokenizer = EsmTokenizer.from_pretrained(f"facebook/{args.esm_model}")

    # Tokenize
    encoded_inputs = [
        tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False)
        for seq in training_data['sequence']
    ]
    labels_list = training_data['neq_class'].tolist()
    
# =============================================================================
#     # Choose cross-validation: stratified is disabled since sequence majority label was not good.
#     if args.cv_type == "stratified":
#         kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
#         splits = kfold.split(training_data, sequence_majority_labels)
#     else:
# =============================================================================
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    splits = kfold.split(training_data)
        
    scaler = GradScaler() if args.mixed_precision else None
     
     
    fold_index = 0
    for train_ids, test_ids in splits:
         fold_index += 1
         print(f"\n========== Fold {fold_index} ==========")

         # Build subsets
         train_enc = [encoded_inputs[i] for i in train_ids]
         train_lbls = [labels_list[i] for i in train_ids]
         val_enc = [encoded_inputs[j] for j in test_ids]
         val_lbls = [labels_list[j] for j in test_ids]

         train_dataset = SequenceClassificationDataset(train_enc, train_lbls)
         val_dataset = SequenceClassificationDataset(val_enc, val_lbls)

# =============================================================================
#          # Oversampling
#          if args.oversampling:
#              train_majority_labels = [sequence_majority_labels[i] for i in train_ids]
#              train_df = pd.DataFrame({
#                  'encoded_inputs': train_enc,
#                  'labels': train_lbls,
#                  'majority_label': train_majority_labels
#              })
#              class_counts = train_df['majority_label'].value_counts()
#              max_count = class_counts.max()
# 
#              oversampled_dfs = []
#              for cls_ in class_counts.index:
#                  subset_df = train_df[train_df['majority_label'] == cls_]
#                  n_samples = max_count - len(subset_df)
#                  if n_samples > 0:
#                      oversampled_subset = subset_df.sample(n=n_samples, replace=True, random_state=42)
#                      oversampled_dfs.append(oversampled_subset)
# 
#              if oversampled_dfs:
#                  oversampled_df = pd.concat([train_df] + oversampled_dfs, ignore_index=True)
#              else:
#                  oversampled_df = train_df
# 
#              oversampled_df = oversampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
#              train_dataset = SequenceClassificationDataset(
#                  list(oversampled_df['encoded_inputs']),
#                  list(oversampled_df['labels'])
#              )
# =============================================================================

         # DataLoaders
         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=lambda b: collate_fn_sequence(b, tokenizer))
         val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=lambda b: collate_fn_sequence(b, tokenizer))

         # -----------------------------
         # Loss Function
         # -----------------------------
         if args.loss_function == "focal":
             if args.focal_class_weights:
                 # Compute class weights from the dataset
                 all_train_labels_flat = [lab for sublab in train_dataset.labels for lab in sublab]
                 unique_classes = np.unique(all_train_labels_flat)
                 cw_np = compute_class_weight('balanced', classes=unique_classes, y=all_train_labels_flat)
                 alpha_tensor = torch.tensor(cw_np, dtype=torch.float).to(device)
                 print(f"Using Focal Loss with alpha (class weights) = {cw_np}.")
             else:
                 alpha_tensor = None
                 print("Using Focal Loss without class weights (alpha=None).")

             loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2, ignore_index=-1)

         else:
             # crossentropy
             loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
             print("Using CrossEntropy Loss.")

         # -----------------------------
         # Initialize Embedding Model
         # -----------------------------
         embedding_model = EsmModel.from_pretrained(f"facebook/{args.esm_model}")
         embedding_model.to(device)
         embedding_model.train()

         # Freeze layers if specified
         if args.freeze_layers:
             # e.g., "0-5" => freeze layers [0..5], unfreeze the rest
             start_layer, end_layer = map(int, args.freeze_layers.split('-'))
             freeze_list = range(start_layer, end_layer + 1)
             for name, param in embedding_model.named_parameters():
                 if "encoder.layer" in name:
                     layer_num = int(name.split('.')[2])
                     if layer_num in freeze_list:
                         param.requires_grad = False
                     else:
                         param.requires_grad = True
                 else:
                     # Typically keep embedding + layer norms unfrozen, but you can decide
                     param.requires_grad = True

         # Choose architecture
         if args.architecture == "bilstm":
             model = BiLSTMClassificationModel(
                 embedding_model=embedding_model,
                 hidden_size=args.hidden_size,
                 num_layers=args.num_layers,
                 num_classes=args.num_classes,
                 dropout=args.dropout
             )
             print("Using BiLSTM architecture.")
         else:
             model = BiLSTMWithSelfAttentionModel(
                 embedding_model=embedding_model,
                 hidden_size=args.hidden_size,
                 num_layers=args.num_layers,
                 num_classes=args.num_classes,
                 dropout=args.dropout
             )
             print("Using BiLSTM + Self-Attention architecture.")

         model.to(device)

         # -----------------------------
         # Optimizer & Scheduler
         # -----------------------------
         optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

         if args.lr_scheduler == "reduce_on_plateau":
             scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
             print("Using ReduceLROnPlateau scheduler.")
         else:
             scheduler = None
             print("No learning rate scheduler used.")

         # -----------------------------
         # Training Loop
         # -----------------------------
         best_val_loss = float('inf')
         epochs_no_improve = 0

         train_losses = []
         val_losses = []

         for epoch in range(1, args.epochs + 1):
             model.train()
             running_loss = 0.0

             for batch in train_loader:
                 input_ids = batch['input_ids'].to(device)
                 attention_mask = batch['attention_mask'].to(device)
                 labels = batch['labels'].to(device)

                 optimizer.zero_grad()

                 if scaler and args.mixed_precision:
                     with autocast():
                         outputs = model(input_ids, attention_mask)
                         outputs_flat = outputs.view(-1, args.num_classes)
                         labels_flat = labels.view(-1)
                         loss = loss_fn(outputs_flat, labels_flat)

                     scaler.scale(loss).backward()
                     scaler.step(optimizer)
                     scaler.update()
                 else:
                     outputs = model(input_ids, attention_mask)
                     outputs_flat = outputs.view(-1, args.num_classes)
                     labels_flat = labels.view(-1)
                     loss = loss_fn(outputs_flat, labels_flat)

                     loss.backward()
                     optimizer.step()

                 running_loss += loss.item()

             avg_train_loss = running_loss / len(train_loader)
             train_losses.append(avg_train_loss)
             print(f"[Fold {fold_index}, Epoch {epoch}] Training Loss: {avg_train_loss:.4f}")

             # Validation
             avg_val_loss = compute_validation_loss(model, val_loader, loss_fn, device, args.num_classes)
             val_losses.append(avg_val_loss)
             print(f"[Fold {fold_index}, Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")

             if scheduler is not None:
                 # Step only if using reduce_on_plateau
                 scheduler.step(avg_val_loss)

             # Early stopping check
             if avg_val_loss < best_val_loss:
                 best_val_loss = avg_val_loss
                 epochs_no_improve = 0
                 best_model_path = os.path.join(run_folder, f"best_model_fold_{fold_index}.pt")
                 torch.save(model.state_dict(), best_model_path)
             else:
                 epochs_no_improve += 1
                 if epochs_no_improve >= args.patience:
                     print("Early stopping!")
                     break

         # Plot Loss Curves
         plt.figure()
         plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
         plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
         plt.xlabel('Epoch')
         plt.ylabel('Loss')
         plt.title(f'Loss Curves - Fold {fold_index}')
         plt.legend()
         loss_curve_path = os.path.join(run_folder, f"loss_curves_fold_{fold_index}.png")
         plt.savefig(loss_curve_path)
         plt.close()

         # Load best model and evaluate
         model.load_state_dict(torch.load(os.path.join(run_folder, f'best_model_fold_{fold_index}.pt')))
         evaluate_model_and_save_plots(model, val_loader, device, fold_index, args.num_classes, run_folder)

    print("\nTraining and evaluation complete.")
    print(f"All outputs saved in folder: {run_folder}")
