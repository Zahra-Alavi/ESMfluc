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
    LSTMWithMultiHeadAttentionModel,
    FocalLoss, 
    LSTMClassificationModel,
    LSTMWithSelfAttentionModel,
    TransformerClassificationModel
)

def tokenize(sequences, tokenizer):
    return [tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False) for seq in sequences]

def compute_validation_loss(model, data_loader, loss_fn, loss_type, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
            
            y_preds = model(input_ids, attention_mask)
            if "bce" in loss_type:
                loss = loss_fn(y_preds.squeeze(-1), y.float())
            else:
                y_preds = y_preds.view(-1, y_preds.shape[-1])
                y = y.view(-1)
                loss = loss_fn(y_preds, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        all_preds, all_targets = [], []
        results = []

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)

            y_preds = model(input_ids, attention_mask)

            if "bce" in loss_fn:
                y_probs = torch.sigmoid(y_preds)
                preds = (y_probs > 0.5).int()

                for i in range(len(batch['sequences'])):
                    seq = batch['sequences'][i]
                    pred = preds[i].cpu().numpy().tolist()
                    true_label = y[i].int().cpu().numpy().tolist()
                    results.append({
                        'sequence': seq,
                        'pred': pred,
                        'true label': true_label
                    })
                    all_preds.append(pred)
                    all_targets.append(true_label)

            else:
                y_probs = torch.softmax(y_preds, dim=-1)
                y_preds = torch.argmax(y_probs, dim=-1)

                for i in range(len(batch['sequences'])):
                    seq = batch['sequences'][i]
                    mask = batch['labels'][i] != -1
                    neq_values = batch['neq_values'][i][mask].tolist()
                    pred = y_preds[i][mask].cpu().numpy().tolist()
                    true_label = y[i][mask].cpu().numpy().tolist()
                    results.append({
                        'sequence': seq,
                        'neq values': neq_values,
                        'pred': pred,
                        'true label': true_label
                    })
                    all_preds.extend(pred)
                    all_targets.extend(true_label)

        # Convert predictions and targets for metrics
        if 'bce' in loss_fn:
            # Flatten for multi-label metrics
            all_preds = [item for sublist in all_preds for item in sublist]
            all_targets = [item for sublist in all_targets for item in sublist]

        report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(all_targets, all_preds)
        results_df = pd.DataFrame(results)
        return report, conf_matrix, results_df

def create_run_folder(folder_name):
    now = datetime.datetime.now()
    results_folder = "../../results/"
    if len(folder_name) == 0:
        folder_name = now.strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = results_folder + folder_name
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print("The result directory is", result_dir)
    return result_dir

    
def get_loss_fn(args, train_dataset):
    if "focal" in args.loss_function:
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
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2, ignore_index=-1, loss_type=args.loss_function.split('-')[-1])
    elif args.loss_function == "ce":
        print("Using CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    elif args.loss_function == "bce":
        print("Using BCEWithLogitsLoss")
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
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

def define_lstm_bidirectional(model_name):
    if "lstm" in model_name:
        if "bilstm" in model_name:
            return True
        else:
            return False
    else:
        raise ValueError(f"Invalid model name: {model_name}. Expected 'lstm' or 'bilstm' in the name.")
    
def determine_num_classes(args):
    # Because we should have num_classes = 1 for binary classificaiton
    if args.num_classes == 2:
        if "bce" in args.loss_function:
            return 1
        else:
            return 2
    else:
        return args.num_classes

def set_up_classification_model(args):
    embedding_model = set_up_embedding_model(args)
    num_classes = determine_num_classes(args)
    if args.architecture == "bilstm" or args.architecture == "lstm":
        print("Using LSTM model without attention")
        bidirectional = define_lstm_bidirectional(args.architecture)
        model = LSTMClassificationModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=num_classes,
            bidirectional=bidirectional
        )
    elif args.architecture == "bilstm_attention" or args.architecture == "lstm_attention":
        print("Using LSTM with SelfAttention model")
        bidirectional = define_lstm_bidirectional(args.architecture)
        model = LSTMWithSelfAttentionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=num_classes, 
            bidirectional=bidirectional
        )
    elif args.architecture == "bilstm_multihead_attention" or args.architecture == "lstm_multihead_attention":
        print("Using LSTM with MultiHeadAttention model")
        bidirectional = define_lstm_bidirectional(args.architecture)
        model = LSTMWithMultiHeadAttentionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=num_classes,
            bidirectional=bidirectional,
        )
    
    elif args.architecture == "transformer":
        print("Using Transformer model")
        model = TransformerClassificationModel(
            embedding_model=embedding_model,
            nhead=args.transformer_nhead,
            num_encoder_layers=args.transformer_num_encoder_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            num_classes=num_classes,
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
    
    run_folder = create_run_folder(args.result_foldername)
        
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
    X_train = train_data["sequence"].tolist()
    X_test = test_data["sequence"].tolist()
    y_label_train = train_data["neq_class"].tolist()
    y_label_test= test_data["neq_class"].tolist()
    y_raw_train = train_data["neq"].tolist()
    y_raw_test = test_data["neq"].tolist()
    
    # If having scheduler as ReduceLROnPlateau, split the data into train and validation
    val_loader = None
    if scheduler:
        X_train, X_val, y_label_train, y_label_val, y_raw_train, y_raw_val = train_test_split(X_train, y_label_train, y_raw_train, test_size=0.2)
        X_val_tokenized = tokenize(X_val, tokenizer)
        val_dataset = SequenceClassificationDataset(X_val_tokenized, y_label_val, X_val, y_raw_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
        
    X_train_tokenized = tokenize(X_train, tokenizer)
    X_test_tokenized = tokenize(X_test, tokenizer)
    train_dataset = SequenceClassificationDataset(X_train_tokenized, y_label_train, X_train, y_raw_train)
    test_dataset = SequenceClassificationDataset(X_test_tokenized, y_label_test, X_test, y_raw_test)
    
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
        
    # -------------------------
    # Training loop
    # -------------------------
    best_model_path = f"{run_folder}/best_model.pth"
    loss_fn = get_loss_fn(args, train_dataset)
    # Perform training only if best_model.pth does not exist
    if os.path.exists(best_model_path):
        print("Model already trained. Skipping training.")
        state_dict = torch.load(best_model_path)
        new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        print("Starting training.....................................")
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
                        if "bce" in args.loss_function:
                            loss = loss_fn(y_preds.squeeze(-1), y.float())
                        else:
                            y_preds_flat = y_preds.view(-1, args.num_classes)
                            y_flat = y.view(-1)
                            loss = loss_fn(y_preds_flat, y_flat)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    y_preds = model(input_ids, attention_mask)
                    if args.loss_type == "bce":
                        loss = loss_fn(y_preds.squeeze(-1), y.float())
                    else:
                        y_preds_flat = y_preds.view(-1, args.num_classes)
                        y_flat = y.view(-1)
                        loss = loss_fn(y_preds_flat, y_flat)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"[Epoch {epoch}] Training Loss: {avg_train_loss:.4f}")
            
            avg_val_loss = compute_validation_loss(model, val_loader, loss_fn, args.loss_function.split("-")[-1], args.device)
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
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
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
    
    # -------------------------
    # Evaluation
    # -------------------------
    print("Evaluating model on test data...")
    cls_report, conf_matrix, results_df = evaluate(model, test_loader, args.loss_function.split("-")[-1], args.device)
    print(cls_report)
    print(conf_matrix)
    
    results_df.to_csv(f"{run_folder}/results.csv", index=False)
    
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
