"""
Created on Tue Feb  4 10:00:07 2025

@author: Ngoc Kim Ngan Tran
"""
# train2.py
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler

from collections import Counter


from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_utils import create_classification_func, load_and_preprocess_data, SequenceClassificationDataset, collate_fn_sequence, compute_sampling_weights

from transformers import EsmModel, EsmTokenizer

from models import (
    FocalLoss, 
    BiLSTMClassificationModel,
    BiLSTMWithSelfAttentionModel,
    TransformerClassificationModel
)

def tokenize(sequences, tokenizer):
    return [tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False) for seq in sequences]

def compute_validation_loss(model, data_loader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            
            y_preds = model(input_ids, attention_mask)
            y_preds = y_preds.view(-1, y_preds.shape[-1])
            y = y.view(-1)
            loss = loss_fn(y_preds, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        all_preds, all_targets = [], []
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            
            y_preds = model(input_ids, attention_mask)
            y_probs = torch.softmax(y_preds, dim=-1)
            y_preds = torch.argmax(y_probs, dim=-1)
            y_preds = y_preds.view(-1)
            y = y.view(-1)
            
            mask_flat = y != -1
            y_preds = y_preds[mask_flat]
            y = y[mask_flat]
            all_preds.extend(y_preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        report = classification_report(all_targets, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_targets, all_preds)
        return report, conf_matrix

def create_run_folder():
    now = datetime.datetime.now()
    folder_name = "../../results/" + now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(folder_name)
    return folder_name

    
def get_loss_fn(args, train_dataset):
    if args.loss_function == "focal":
        if args.focal_class_weights:
            print("Computing class weights for FocalLoss")
            # Compute class weights from the dataset
            all_train_labels_flat = [lab for sublab in train_dataset.labels for lab in sublab]
            class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels_flat), y=all_train_labels_flat)
            alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(args.device)
            print(f"Using Focal Loss with alpha (class weights) = {class_weights}")
        else:
            alpha_tensor = None
            print("Using FocalLoss without class weights")
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2, ignore_index=-1)
    else:
        print("Using CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    return loss_fn

def set_up_embedding_model(args):
    embedding_model = EsmModel.from_pretrained(f"facebook/{args.esm_model}")
    embedding_model.to(args.device)
    embedding_model.train()
    
    # Free layers
    if args.freeze_layers:
        # Ex: '0-5' means freeze layers 0..5, and unfreeze the rest
        start_layer, end_layer = map(int, args.freeze_layers.split("-"))
        freeze_list = range(start_layer, end_layer+1)
        for name, param in embedding_model.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])
                param.requires_grad = not layer_num in freeze_list
            else:   
                param.requires_grad = True
        print(f"Freezing layers {args.freeze_layers}")
    return embedding_model

def set_up_classification_model(args):
    embedding_model = set_up_embedding_model(args)
    if args.architecture == "bilstm":
        print("Using BiLSTM model")
        model = BiLSTMClassificationModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=args.num_classes
        )
    elif args.architecture == "bilstm_attention":
        print("Using BiLSTM with SelfAttention model")
        model = BiLSTMWithSelfAttentionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=args.num_classes
        )
    
    elif args.architecture == "transformer":
        print("Using Transformer model")
        model = TransformerClassificationModel(
            embedding_model=embedding_model,
            nhead=args.transformer_nhead,
            num_encoder_layers=args.transformer_num_encoder_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            num_classes=args.num_classes,
            dropout=args.dropout
        )   
    
    else:
        raise ValueError(f"Invalid architecture: {args.architecture}")
        
        
    model.to(args.device)
    
    # Wrap in DataParallel for bigger models
    if args.data_parallel and torch.cuda.device_count() > 1:
       print(f"Using nn.DataParallel on {torch.cuda.device_count()} GPUs")
       model = nn.DataParallel(model)
       
    return model
     
def train(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_folder = create_run_folder()
        
    model = set_up_classification_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    if args.lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        print("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = None
        print("Not using any scheduler")
    
    # Load data
    labeled_neq = create_classification_func(args.num_classes, args.neq_thresholds)
    train_data = load_and_preprocess_data(args.train_data_file, labeled_neq)
    test_data = load_and_preprocess_data(args.test_data_file, labeled_neq)

    # Preprocessing data
    tokenizer = EsmTokenizer.from_pretrained(f"facebook/{args.esm_model}")
    X_train = tokenize(train_data['sequence'], tokenizer) # [input_ids, attention_mask]
    X_test = tokenize(test_data['sequence'], tokenizer)
    y_train = train_data['neq_class'].tolist()
    y_test = test_data['neq_class'].tolist()
    
    
    # If having scheduler as ReduceLROnPlateau, split the data into train and validation
    val_loader = None
    if scheduler:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        val_dataset = SequenceClassificationDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
        
        
    train_dataset = SequenceClassificationDataset(X_train, y_train)
    test_dataset = SequenceClassificationDataset(X_test, y_test)
    
    # Check original class distribution before oversampling
    raw_labels = []
    for i in range(len(train_dataset)):
        raw_labels.extend(train_dataset.labels[i])

    raw_class_counts = Counter(raw_labels)
    print("Original Class Distribution:", raw_class_counts)
    
    # If oversampling is enabled, compute weights
    if args.oversampling:
       print("Applying oversampling using WeightedRandomSampler...")
       sampling_weights = compute_sampling_weights(
           train_dataset, 
           num_classes=args.num_classes,
           neq_thresholds=args.neq_thresholds,
           oversampling_threshold=args.oversampling_threshold, 
           undersampling_threshold=args.undersampling_threshold,
           undersampling_intensity = args.undersampling_intensity,
           oversampling_intensity = args.oversampling_intensity
       )
       
       sampler = WeightedRandomSampler(
           weights=sampling_weights,
           num_samples=len(train_dataset),
           replacement=True  # Allows oversampling
       )
       
       train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
       
       # Collect labels from the sampled data in the DataLoader
       oversampled_labels = []
       for batch in train_loader:
           batch_labels = batch['labels'].cpu().numpy().flatten()
           batch_labels = batch_labels[batch_labels != -1]  # Remove padding values
           oversampled_labels.extend(batch_labels)

       oversampled_class_counts = Counter(oversampled_labels)
       print("Sampled Class Distribution After Oversampling:", oversampled_class_counts)


    else:
       train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
       
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    
    
    
    loss_fn = get_loss_fn(args, train_dataset)
        
    # -------------------------
    # Training loop
    # -------------------------
    scaler = GradScaler(device=args.device) if args.mixed_precision else None
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    val_losses = []
    train_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y = batch['labels'].to(args.device)
            
            optimizer.zero_grad()
            
            if scaler and args.mixed_precision:
                with torch.amp.autocast(device_type=args.device):
                    y_preds = model(input_ids, attention_mask)
                    y_preds_flat = y_preds.view(-1, args.num_classes)
                    y_flat = y.view(-1)
                    loss = loss_fn(y_preds_flat, y_flat)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y_preds = model(input_ids, attention_mask)
                y_preds_flat = y_preds.view(-1, args.num_classes)
                y_flat = y.view(-1)
                loss = loss_fn(y_preds_flat, y_flat)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch}] Training Loss: {avg_train_loss:.4f}")
        
        avg_val_loss = compute_validation_loss(model, val_loader, loss_fn, args.device)
        val_losses.append(avg_val_loss)
        print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")
        
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            if isinstance(model, nn.DataParallel):
                # Save the best model
                torch.save(model.module.state_dict(), f"{run_folder}/best_model.pth")
            else:
                torch.save(model.state_dict(), f"{run_folder}/best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
       
            
    # Plot loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(f"{run_folder}/loss_curve.png")
    
    # Evaluation
    cls_report, conf_matrix = evaluate(model, test_loader, loss_fn, args.device)
    print(cls_report)
    print(conf_matrix)
    
    # Save classification report and confusion matrix
    with open(f"{run_folder}/classification_report.txt", "w") as f:
        f.write(str(cls_report))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(f"{run_folder}/confusion_matrix.png")
    plt.close()
    
    # Save args to text
    with open(f"{run_folder}/args.txt", "w") as f:
        f.write(str(args))
        
    print("Training completed")