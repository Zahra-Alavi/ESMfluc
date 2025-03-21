"""
Created on Tue Feb  4 10:00:07 2025

@author: Ngoc Kim Ngan Tran
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.amp import GradScaler


from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_utils import create_classification_func, load_and_preprocess_data, SequenceClassificationDataset, collate_fn_sequence

from transformers import EsmModel, EsmTokenizer
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk import client
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)

from models import (
    FocalLoss, 
    BiLSTMClassificationModel,
    BiLSTMWithSelfAttentionModel
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
    if args.esm_model.startswith("esmc"):
        embedding_model = ESMC.load_model(args.esm_model)
        embedding_model.to(args.device)
        embedding_model.train()
    else:
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
    else:
        raise ValueError(f"Invalid architecture: {args.architecture}")
    model.to(args.device)
    return model

def embed_sequence(model: ESM3InferenceClient, sequence: str) -> LogitsOutput:
    EMBEDDING_CONFIG = LogitsConfig(
        sequence=True, return_embeddings=True, return_hidden_states=True
    )
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    return output.embeddings

def batch_embed(
    model: ESM3InferenceClient, inputs: Sequence[ProteinType]
) -> Sequence[LogitsOutput]:
    """Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(embed_sequence, model, protein) for protein in inputs
        ]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
    return results

def train(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    labeled_neq = create_classification_func(args.num_classes, args.neq_thresholds)
    train_data = load_and_preprocess_data(args.train_data_file, labeled_neq)
    test_data = load_and_preprocess_data(args.test_data_file, labeled_neq)

    # Preprocessing data
    if not args.esm_model.startswith("esmc"):
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/{args.esm_model}")
        X_train = tokenize(train_data['sequence'], tokenizer) # [input_ids, attention_mask]
        X_test = tokenize(test_data['sequence'], tokenizer)
        y_train = train_data['neq_class'].tolist()
        y_test = test_data['neq_class'].tolist()
        
        train_dataset = SequenceClassificationDataset(X_train, y_train)
        test_dataset = SequenceClassificationDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
    else:
        # ESMC model
        
        # the protein sequence need to switch Protein
        token = "4cNSB4g1pJ7xgNtvOcrp4Q"
        model = client(
            model="esmc-300m-2024-12", url="https://forge.evolutionaryscale.ai", token=token
        )
        ESMC_6B_EMBEDDING_CONFIG = LogitsConfig(return_hidden_states=True, ith_hidden_layer=55)
        #This X_train doesn't have attention_mask or input_ids
        X_train = batch_embed(model, X_train["sequence"].tolist())
        X_test = batch_embed(model, X_test["sequence"].tolist())
        y_train = train_data['neq_class'].tolist()
        y_test = test_data['neq_class'].tolist()
        
        
    
    model = set_up_classification_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    if args.lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        print("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = None
        print("Not using any scheduler")

    # If having scheduler as ReduceLROnPlateau, split the data into train and validation
    val_loader = None
    if scheduler:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        val_dataset = SequenceClassificationDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_sequence(x, tokenizer))
        
    train_dataset = SequenceClassificationDataset(X_train, y_train)
    test_dataset = SequenceClassificationDataset(X_test, y_test)
    
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
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
       
            
    # Plot loss curve
    run_folder = create_run_folder()
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
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
    # Save args to text
    with open(f"{run_folder}/args.txt", "w") as f:
        f.write(str(args))
    with open(f"{run_folder}/classification_report.txt", "w") as f:
        f.write(str(cls_report))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(f"{run_folder}/confusion_matrix.png")
    plt.close()
    
    print("Training completed")
