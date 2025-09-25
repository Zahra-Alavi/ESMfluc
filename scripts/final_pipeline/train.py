"""
Created on Tue Feb  4 10:00:07 2025

"""

import os
import datetime
import time, json
try:
    import psutil
except ImportError:
    psutil = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from collections import Counter


from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_utils import create_classification_func, load_and_preprocess_data, SequenceClassificationDataset, collate_fn_sequence, compute_sampling_weights

from transformers import EsmModel, EsmTokenizer

from models import (
    FocalLoss, 
    BiLSTMClassificationModel,
    BiLSTMWithSelfAttentionModel,
    TransformerClassificationModel,
    ESMLinearTokenClassifier
)

from nc_losses import NCLoss



def tokenize(sequences, tokenizer):
    return [tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False) for seq in sequences]

def compute_validation_loss(model, data_loader, criterion, args):
    if data_loader is None:
        raise ValueError("Validation requested but val_loader is None.")
    m = model.module if isinstance(model, nn.DataParallel) else model

    N = len(data_loader)
    total_loss = val_nc1 = val_nc2 = val_sup = 0.0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y              = batch['labels'].to(args.device)

            logits, feats = m(input_ids, attention_mask,
                                  return_features=("pre" if args.head=="softmax" else
                                                   "post" if args.head=="postfc" else "pre"))

            logits_flat = None if logits is None else logits.reshape(-1, args.num_classes)
            feats_flat  = feats.reshape(-1, feats.size(-1))
            y_flat      = y.reshape(-1)

            loss, (sup, nc1, nc2, _, _) = criterion(logits_flat, y_flat, feats_flat) \
                if hasattr(criterion, "NC1") else (criterion(logits_flat, y_flat), (torch.tensor(0., device=args.device),)*3 + (None, None))

            total_loss += loss.item()
            val_sup    += sup.item()
            val_nc1    += nc1.item()
            val_nc2    += nc2.item()

    avg_loss = total_loss / max(N, 1)
    sup_loss = val_sup / max(N, 1)
    nc1_loss = val_nc1 / max(N, 1)
    nc2_loss = val_nc2 / max(N, 1)
    return avg_loss, sup_loss, nc1_loss, nc2_loss


def evaluate(model, data_loader, criterion, args):
    
    m = model.module if isinstance(model, nn.DataParallel) else model 
    
    with torch.no_grad():
        m.eval()
        all_preds, all_targets = [], []

        for batch in data_loader:
            input_ids      = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y              = batch['labels'].to(args.device)

            logits, feats = m(input_ids, attention_mask,
                                  return_features = ("pre"  if args.head=="softmax"
                                                     else "post" if args.head=="postfc"
                                                     else "pre"))

            if args.head == "centroid":
                # nearest-centre classification
                preds = centroid_predict(feats, criterion.NC1.means)
            else:
                probs = torch.softmax(logits, -1)
                preds = probs.argmax(-1)

            preds = preds.reshape(-1)
            y     = y.reshape(-1)
            mask  = y != -1
            all_preds.extend(preds[mask].cpu().numpy())
            all_targets.extend(y[mask].cpu().numpy())

        report      = classification_report(all_targets, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_targets, all_preds)
        return report, conf_matrix


def create_run_folder(desired_folder_name):
    now = datetime.datetime.now()
    folder_name = "./results/"
    if desired_folder_name != "timestamp":
        folder_name += desired_folder_name
    else:
        folder_name += now.strftime("%Y-%m-%d-%H-%M-%S")
    if not os.path.exists(folder_name):
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
    
    if getattr(args, "freeze_all_backbone", False):
        for p in embedding_model.parameters():
            p.requires_grad = False
        embedding_model.eval()    #disable dropout in the frozen backbone
        print("Frozen ALL ESM parameters.")
        n_trainable = sum(p.requires_grad for p in embedding_model.parameters())
        print(f"Trainable ESM params: {n_trainable}")
        return embedding_model
    
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
        
    n_trainable = sum(p.requires_grad for p in embedding_model.parameters())
    print(f"Trainable ESM params: {n_trainable}") 
        
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
            num_classes=args.num_classes,
            head = args.head, 
            bidirectional=args.bidirectional       
        )
    elif args.architecture == "bilstm_attention":
        print("Using BiLSTM with SelfAttention model")
        model = BiLSTMWithSelfAttentionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=args.num_classes,
            head = args.head,
            bidirectional=args.bidirectional
        )
    
    elif args.architecture == "transformer":
        print("Using Transformer model")
        model = TransformerClassificationModel(
            embedding_model=embedding_model,
            nhead=args.transformer_nhead,
            num_encoder_layers=args.transformer_num_encoder_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            num_classes=args.num_classes,
            dropout=args.dropout,
            head = args.head
        )
        
    elif args.architecture == "esm_linear":
        print("Using ESM-only linear token classifier")
        model = ESMLinearTokenClassifier(
            embedding_model=embedding_model,
            num_classes=args.num_classes,
            head=args.head,   
        )
    
    else:
        raise ValueError(f"Invalid architecture: {args.architecture}")
        
    if args.head == "centroid" and hasattr(model, "fc"):
        print("Removing FC layer for centroid mode")
        del model.fc
        model.use_fc = False
    else:
        model.use_fc = True
        
        
    model.to(args.device)
    
    # Wrap in DataParallel for bigger models
    if args.data_parallel and torch.cuda.device_count() > 1:
       print(f"Using nn.DataParallel on {torch.cuda.device_count()} GPUs")
       model = nn.DataParallel(model)
       
    return model

def centroid_predict(feats, centres):
    B,L,D = feats.shape
    feats  = torch.nn.functional.normalize(feats.reshape(-1,D), p=2, dim=1)
    centres= torch.nn.functional.normalize(centres, p=2, dim=1)
    cos    = torch.matmul(feats, centres.t())                 # (B·L,K)
    return cos.argmax(-1).reshape(B,L)

def build_nc_criterion(args, feat_dim, occurrence_list, weight_factor):
    def sup_name():
        if args.loss_function == "focal":
            return FocalLoss(ignore_index=-1)
        else:
            return nn.CrossEntropyLoss(ignore_index=-1)
    
    # decide lambda_CE by loss_mode and head
    if args.loss_mode == "nc" or args.head == "centroid":
        lambda_ce = 0.0                    # no supervised term in pure-NC or centroid
    elif args.loss_mode == "both":
        lambda_ce = args.lambda_ce         # combine NC + supervised
    else:
        lambda_ce = 0.0                    # supervised-only won’t come here
    
    crit = NCLoss(
        sup_criterion = sup_name,
        lambda_CE     = lambda_ce,
        lambda1       = args.lambda_nc1,
        lambda2       = args.beta_nc2,
        nc1           = "NC1Loss_v5_cosine",
        nc2           = "NC2Loss",
        num_classes   = args.num_classes,
        feat_dim      = feat_dim,
        device        = args.device,
        occurrence_list = occurrence_list,
        weight_factor   = weight_factor
    )
    return crit

def compute_sup_nc_loss(criterion, logits_flat, y_flat, feats_flat):
    if isinstance(criterion, NCLoss):
        loss, (sup, nc1, nc2, _, _) = criterion(logits_flat, y_flat, feats_flat)
        return loss, sup, nc1, nc2
    else:
        loss = criterion(logits_flat, y_flat)  # CE/Focal only
        z = torch.zeros((), device=logits_flat.device, dtype=loss.dtype)
        return loss, z, z, z
    
    
def infer_feat_dim(model, args):
    if args.head == "postfc":
        return args.num_classes
    m = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(m, "output_dim"):
        return m.output_dim
    if args.architecture == "transformer":
        return m.embedding_model.config.hidden_size
    raise ValueError("Cannot infer feature dimension.")
     
def train(args):
    
    run_folder = create_run_folder(args.result_foldername)
        
    model = set_up_classification_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    dp = bool(args.data_parallel) and torch.cuda.device_count() > 1
        
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
    need_val = (args.lr_scheduler == "reduce_on_plateau") or (args.patience and args.patience > 0)

    if need_val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=None, random_state=args.seed)
        val_dataset = SequenceClassificationDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: collate_fn_sequence(x, tokenizer),)
            
        
        
    train_dataset = SequenceClassificationDataset(X_train, y_train)
    test_dataset = SequenceClassificationDataset(X_test, y_test)
    
    # Check original class distribution before oversampling
    raw_labels = []
    for i in range(len(train_dataset)):
        raw_labels.extend(train_dataset.labels[i])

    raw_class_counts = Counter(raw_labels)
    print("Original Class Distribution:", raw_class_counts)
    
    flat = [lab for seq in train_dataset.labels for lab in seq]
    occurrence_list = [flat.count(k) for k in range(args.num_classes)]
    weight_factor = torch.tensor(
        [1.0/np.sqrt(max(n,1)) for n in occurrence_list],  # avoid /0
        dtype=torch.float, device=args.device)
    
    
    
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
           replacement=True,  # Allows oversampling
       )
       
       train_loader = DataLoader(train_dataset, 
                                 batch_size=args.batch_size, 
                                 sampler=sampler, 
                                 collate_fn=lambda x: collate_fn_sequence(x, tokenizer), 
                                 drop_last=dp,
                                )
       
       # Collect labels from the sampled data in the DataLoader
       oversampled_labels = []
       for batch in train_loader:
           batch_labels = batch['labels'].cpu().numpy().flatten()
           batch_labels = batch_labels[batch_labels != -1]  # Remove padding values
           oversampled_labels.extend(batch_labels)

       oversampled_class_counts = Counter(oversampled_labels)
       print("Sampled Class Distribution After Oversampling:", oversampled_class_counts)


    else:
       train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn_sequence(x, tokenizer), drop_last=dp)
       
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_sequence(x, tokenizer), drop_last=False)
    
    # --- choose criterion by loss_mode ---
    
    if args.loss_mode == "supervised":
        # pure CE/Focal
        criterion = get_loss_fn(args, train_dataset)

    else:
        # NC only or NC + supervised
        feat_dim = infer_feat_dim(model, args)
        criterion = build_nc_criterion(
            args, feat_dim=feat_dim,
            occurrence_list=occurrence_list,
            weight_factor=weight_factor)

    # sanity: centroid head requires NC (no logits for CE/Focal)
    if args.loss_mode == "supervised" and args.head == "centroid":
        raise ValueError("centroid head has no logits: use --loss_mode nc or --loss_mode both.")

    
    
        
    # -------------------------
    # Training loop
    # -------------------------
    
    
    use_cuda   = (isinstance(args.device, torch.device) and args.device.type == "cuda") \
             or (isinstance(args.device, str) and args.device == "cuda")
    amp_enabled = bool(args.mixed_precision and use_cuda)
    amp_dtype   = torch.bfloat16 if getattr(args, "amp_dtype", "fp16") == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled)   # no device kwarg
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    val_losses = []
    train_losses = []
    
    
    # timing and memory metrics
    run_start = time.perf_counter()
    epoch_times = []
    gpu_epoch_peaks = []
    cpu_rss_start = psutil.Process().memory_info().rss if psutil else None

    on_cuda = (isinstance(args.device, torch.device) and args.device.type == "cuda") or \
              (isinstance(args.device, str) and args.device.startswith("cuda"))
    if on_cuda:
              torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        model.train()
        total_loss = 0
        sum_sup = sum_nc1 = sum_nc2 = 0.0

        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y = batch['labels'].to(args.device)
            
            optimizer.zero_grad()
            
            if amp_enabled and args.device.startswith("cuda"):
                with autocast(dtype=amp_dtype):
                    logits, feats = model(input_ids, attention_mask,
                              return_features=("pre" if args.head=="softmax"
                                               else "post" if args.head=="postfc"
                                               else "pre"))
                    logits_flat = None if logits is None else logits.reshape(-1, args.num_classes)
                    feats_flat  = feats.reshape(-1, feats.size(-1))
                    y_flat      = y.reshape(-1)
                    loss, sup, nc1, nc2 = compute_sup_nc_loss(criterion, logits_flat, y_flat, feats_flat)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, feats = model(input_ids, attention_mask,
                          return_features=("pre" if args.head=="softmax"
                                           else "post" if args.head=="postfc"
                                           else "pre"))
                logits_flat = None if logits is None else logits.reshape(-1, args.num_classes)
                feats_flat  = feats.reshape(-1, feats.size(-1))
                y_flat      = y.reshape(-1)
                loss, sup, nc1, nc2 = compute_sup_nc_loss(criterion, logits_flat, y_flat, feats_flat)
                loss.backward()
                optimizer.step()

            
            total_loss += loss.item()
            sum_sup  += sup.item()
            sum_nc1 += nc1.item()
            sum_nc2 += nc2.item()

            N = len(train_loader)
        avg_train_loss = total_loss / N
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch}] Training Loss: {avg_train_loss:.4f} ⟨sup⟩={sum_sup/N:.4f}  ⟨nc1⟩={sum_nc1/N:.4f}  ⟨nc2⟩={sum_nc2/N:.4f}")
        
        if val_loader is not None:
            avg_val_loss, val_sup_loss, val_nc1_loss, val_nc2_loss = compute_validation_loss(model, val_loader, criterion, args)
            val_losses.append(avg_val_loss)
            print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}, ⟨sup⟩={val_sup_loss:.4f} ⟨nc1⟩={val_nc1_loss:.4f} ⟨nc2⟩={val_nc2_loss:.4f}")

            if scheduler:
                scheduler.step(avg_val_loss)

            # Early stopping
            MIN_DELTA = 1e-3  #small tolerance 
            improved = (best_val_loss - avg_val_loss) > MIN_DELTA
            if improved:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                           f"{run_folder}/best_model.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            # No validation: just save latest each epoch (or only at the end if you prefer)
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                       f"{run_folder}/last_model.pth")
 
            
            
        if args.warmup_epochs and epoch == args.warmup_epochs - 1:
            if args.head in ["centroid", "postfc"]:
                criterion.set_lambda_CE(0.0)
                print("CE component turned off – continuing with NC-only.")
                
        if on_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_time)

    if on_cuda:
        gpu_epoch_peaks.append(int(torch.cuda.max_memory_allocated()))

       
            
    # Plot loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    if len(val_losses) > 0:
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend()
    plt.savefig(f"{run_folder}/loss_curve.png")

    
    # Evaluation
    cls_report, conf_matrix = evaluate(model, test_loader, criterion, args)
    cls_report_df = pd.DataFrame(cls_report).transpose()
    latex_table = cls_report_df.to_latex(float_format="%.2f")
    print(cls_report)
    print(latex_table)
    print(conf_matrix)
    
    # Save classification report and confusion matrix
    with open(f"{run_folder}/classification_report.txt", "w") as f:
        f.write(str(cls_report))
    with open(f"{run_folder}/classification_report.tex", "w") as f:
        f.write(latex_table)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(f"{run_folder}/confusion_matrix.png")
    plt.close()
    
    # Save args to text
    with open(f"{run_folder}/args.txt", "w") as f:
        f.write(str(args))
        
    if on_cuda:
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - run_start
    gpu_overall_peak = int(torch.cuda.max_memory_allocated()) if on_cuda else None
    cpu_rss_end = psutil.Process().memory_info().rss if psutil else None

    metrics = {
        "device": str(args.device),
        "seed": args.seed,
        "amp_enabled": bool(args.mixed_precision and on_cuda),
        "total_seconds": total_seconds,
        "epoch_seconds": epoch_times,
        "gpu_peak_bytes_per_epoch": gpu_epoch_peaks if on_cuda else None,
        "gpu_overall_peak_bytes": gpu_overall_peak,
        "cpu_rss_start_bytes": cpu_rss_start,
        "cpu_rss_end_bytes": cpu_rss_end,
        "epochs_ran": len(epoch_times),
        }
    with open(f"{run_folder}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {run_folder}/metrics.json")
    
    # ---------- RUN SUMMARY CSV (one row) ----------
    # Build a flat dict with config, metrics, report, and confusion matrix

    row = {
        "run_dir": run_folder,
        "batch_size": args.batch_size,
        "embedding_model": args.esm_model,
        "model_architecture": args.architecture,

        # Loss description (explicit fields + compact summary)
        "loss_mode": args.loss_mode,                # "supervised" | "nc" | "both"
        "loss_function": args.loss_function,        # "focal" | "crossentropy"
        "head": args.head,                          # "softmax" | "postfc" | "centroid"
        "loss_desc": f"{args.loss_mode}|{args.loss_function}|head={args.head}",

        # Timing & memory
        "total_time_seconds": total_seconds,
        "epochs_ran": len(epoch_times),
        "gpu_overall_peak_bytes": gpu_overall_peak,
        "cpu_rss_start_bytes": cpu_rss_start,
        "cpu_rss_end_bytes": cpu_rss_end,
        "cpu_rss_delta_bytes": (
            int(cpu_rss_end - cpu_rss_start)
            if (cpu_rss_end is not None and cpu_rss_start is not None) else None
            ),

        # Repro / runtime
        "seed": getattr(args, "seed", None),
        "device": str(args.device),
        "amp_enabled": bool(args.mixed_precision and on_cuda),
        }

    # Flatten classification_report into columns
    # e.g., adds: accuracy, 0_precision, 0_recall, ..., macro_avg_f1_score, weighted_avg_support, etc.
    for key, val in cls_report.items():
        if key == "accuracy":
            row["accuracy"] = val
            continue
        if isinstance(val, dict):
           key_safe = str(key).replace(" ", "_")              # "macro avg" -> "macro_avg"
           for subk, subval in val.items():
               subk_safe = subk.replace("-", "_")             # "f1-score" -> "f1_score"
               row[f"{key_safe}_{subk_safe}"] = subval

    # Flatten confusion matrix into columns
    # cm_true_i_pred_j for every cell
    cm = conf_matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            row[f"cm_true_{i}_pred_{j}"] = int(cm[i, j])

    # Make DataFrame and save
    summary_df = pd.DataFrame([row])
    summary_csv_path = f"{run_folder}/run_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved run summary to {summary_csv_path}")
# ----------------------------------------------

        
    print("Training completed")
