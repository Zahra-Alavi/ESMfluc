"""Training loop for the published binary classification conditions."""

import os
import datetime
import time, json
import re
import random
try:
    import psutil
except ImportError:
    psutil = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from collections import Counter


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from esmfluc.training.data_utils import create_classification_func, load_and_preprocess_data, SequenceClassificationDataset, collate_fn_sequence

from transformers import EsmModel, EsmTokenizer


def assert_exact_sequence_disjoint(**datasets):
    """Fail fast if any exact sequence occurs in more than one named split."""
    sequence_sets = {
        name: set(frame["sequence"].astype(str))
        for name, frame in datasets.items()
        if frame is not None
    }
    names = list(sequence_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = sequence_sets[left] & sequence_sets[right]
            if overlap:
                raise ValueError(
                    f"Data leakage: {len(overlap)} exact sequence(s) occur in both "
                    f"{left} and {right}. Use the fixed grouped split files."
                )

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

from esmfluc.training.models import (
    FocalLoss,
    BiLSTMWithSelfAttentionModel,
    ESMLinearTokenClassifier,
    ESM3Wrapper,
)


def set_global_seed(seed):
    """Configure reproducible training for a fixed software/hardware stack.

    Call this before constructing models or DataLoaders so the seed controls
    task-layer initialization, dropout, shuffling, sampling, and CUDA kernels.
    Deterministic algorithms fail loudly instead of silently using a known
    nondeterministic operation.
    """
    seed = int(seed)

    # cuBLAS requires this setting before CUDA context initialization for
    # deterministic matrix multiplication on CUDA 10.2 and newer.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # The training runner also exports this before Python starts. Setting it
    # here records the intended value for direct main.py/train.py use.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    print(
        f"Reproducibility configured: seed={seed}, "
        "deterministic_algorithms=True, cudnn_benchmark=False"
    )


def load_esm_tokenizer(model_name):
    """Load tokenizer - supports both ESM2 and ESM3"""
    if "esm3" in model_name.lower():
        if not ESM3_AVAILABLE:
            raise ImportError("ESM3 models require the 'esm' library. Install with: pip install esm")
        return EsmSequenceTokenizer()
    else:
        return EsmTokenizer.from_pretrained(f"facebook/{model_name}")

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


def tokenize(sequences, tokenizer):
    return [tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False) for seq in sequences]

def compute_validation_loss(model, data_loader, criterion, args):
    if data_loader is None:
        raise ValueError("Validation requested but val_loader is None.")
    m = model.module if isinstance(model, nn.DataParallel) else model

    N = len(data_loader)
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y              = batch['labels'].to(args.device)

            logits, feats = m(input_ids, attention_mask, return_features="pre")
            logits_flat = logits.reshape(-1, args.num_classes)
            y_flat      = y.reshape(-1)
            loss = criterion(logits_flat, y_flat)
            total_loss += loss.item()

    avg_loss = total_loss / max(N, 1)
    return avg_loss, avg_loss, 0.0, 0.0


def evaluate(model, data_loader, criterion, args):

    m = model.module if isinstance(model, nn.DataParallel) else model

    with torch.no_grad():
        m.eval()
        all_preds, all_targets = [], []

        for batch in data_loader:
            input_ids      = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y              = batch['labels'].to(args.device)

            logits, feats = m(input_ids, attention_mask, return_features="pre")
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
    return FocalLoss(alpha=None, gamma=2, ignore_index=-1)

def parse_layer_range(layer_range):
    start_layer, end_layer = map(int, layer_range.split("-"))
    if end_layer < start_layer:
        raise ValueError(
            f"Invalid --freeze_layers '{layer_range}': end must be >= start."
        )
    return range(start_layer, end_layer + 1)

def esm2_layer_index(name):
    match = re.search(r"(?:^|\.)encoder\.layer\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None

def esm3_layer_index(name):
    """
    Return the ESM3 transformer block index from a parameter name.
    Handles common esm package layouts such as:
      esm3.transformer.blocks.0.*
      esm3.transformer.layers.0.*
      transformer.blocks.0.*
      trunk.blocks.0.*
    """
    patterns = (
        r"(?:^|\.)transformer\.(?:blocks|layers)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:blocks|layers)\.(\d+)(?:\.|$)",
        r"(?:^|\.)trunk\.(?:blocks|layers)\.(\d+)(?:\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None

def apply_partial_backbone_freeze(embedding_model, layer_range, is_esm3):
    freeze_list = set(parse_layer_range(layer_range))
    get_layer_index = esm3_layer_index if is_esm3 else esm2_layer_index
    matched_layers = set()

    for name, param in embedding_model.named_parameters():
        layer_idx = get_layer_index(name)
        if layer_idx is None:
            param.requires_grad = True
            continue
        matched_layers.add(layer_idx)
        param.requires_grad = layer_idx not in freeze_list

    model_label = "ESM3" if is_esm3 else "ESM2"
    if matched_layers:
        unmatched_requested = sorted(freeze_list - matched_layers)
        if unmatched_requested:
            raise ValueError(
                f"--freeze_layers {layer_range} requested nonexistent {model_label} "
                f"layers {unmatched_requested}; matched range is "
                f"{min(matched_layers)}-{max(matched_layers)}."
            )
        frozen_matched = sorted(matched_layers & freeze_list)
        trainable_matched = sorted(matched_layers - freeze_list)
        print(f"Freezing {model_label} layers {layer_range}")
        print(f"  Matched backbone layers: {min(matched_layers)}-{max(matched_layers)}")
        print(f"  Frozen matched layers: {frozen_matched}")
        print(f"  Trainable matched layers: {trainable_matched}")
    else:
        sample_names = [name for name, _ in list(embedding_model.named_parameters())[:12]]
        print(f"WARNING: --freeze_layers was set to {layer_range}, but no {model_label} "
              "transformer layer parameters matched known naming patterns.")
        print(f"  Sample parameter names: {sample_names}")

def keep_frozen_backbone_eval(model):
    m = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(m, "embedding_model"):
        m.embedding_model.eval()

def set_up_embedding_model(args):
    embedding_model, model_type = load_esm_model(args.esm_model, device=args.device)
    print(f"Loaded {model_type} model: {args.esm_model}")

    is_esm3 = "esm3" in args.esm_model.lower()

    if getattr(args, "freeze_all_backbone", False):
        for p in embedding_model.parameters():
            p.requires_grad = False
        embedding_model.eval()    # disable dropout in the frozen backbone
        print("Frozen ALL ESM parameters.")
        n_trainable = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
        print(f"Trainable ESM params: {n_trainable}")
        return embedding_model

    embedding_model.train()

    # Partial layer freezing. Non-transformer parameters remain trainable.
    if args.freeze_layers:
        apply_partial_backbone_freeze(embedding_model, args.freeze_layers, is_esm3)

    n_trainable = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    print(f"Trainable ESM params: {n_trainable}")

    return embedding_model

def set_up_classification_model(args):
    embedding_model = set_up_embedding_model(args)
    if args.architecture == "bilstm_attention":
        print("Using BiLSTM with SelfAttention model")
        model = BiLSTMWithSelfAttentionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=args.num_classes,
            bidirectional=args.bidirectional
        )

    elif args.architecture == "esm_linear":
        print("Using ESM-only linear token classifier")
        model = ESMLinearTokenClassifier(
            embedding_model=embedding_model,
            num_classes=args.num_classes
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
    set_global_seed(args.seed)

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
    val_data = (
        load_and_preprocess_data(args.validation_data_file, labeled_neq)
        if args.validation_data_file else None
    )
    assert_exact_sequence_disjoint(train=train_data, validation=val_data, test=test_data)

    # Preprocessing data
    tokenizer = load_esm_tokenizer(args.esm_model)
    X_train = tokenize(train_data['sequence'], tokenizer) # [input_ids, attention_mask]
    X_test = tokenize(test_data['sequence'], tokenizer)
    y_train = train_data['neq_class'].tolist()
    y_test = test_data['neq_class'].tolist()


    # If having scheduler as ReduceLROnPlateau, split the data into train and validation
    val_loader = None
    need_val = (args.lr_scheduler == "reduce_on_plateau") or (args.patience and args.patience > 0)

    if need_val:
        if not args.validation_data_file:
            raise ValueError(
                "--validation_data_file is required when validation is used. "
                "Validation membership must not be derived from the training seed."
            )
        X_val = tokenize(val_data['sequence'], tokenizer)
        y_val = val_data['neq_class'].tolist()
        val_dataset = SequenceClassificationDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: collate_fn_sequence(x, tokenizer),)



    train_dataset = SequenceClassificationDataset(X_train, y_train)
    test_dataset = SequenceClassificationDataset(X_test, y_test)

    # Report the original class distribution.
    raw_labels = []
    for i in range(len(train_dataset)):
        raw_labels.extend(train_dataset.labels[i])

    raw_class_counts = Counter(raw_labels)
    print("Original Class Distribution:", raw_class_counts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn_sequence(x, tokenizer),
        drop_last=dp,
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_sequence(x, tokenizer), drop_last=False)

    # --- choose criterion: always supervised ---
    criterion = get_loss_fn(args, train_dataset)




    # Training loop


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

    accum_steps = max(1, getattr(args, 'gradient_accumulation_steps', 1))
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps "
              f"(effective batch size = {args.batch_size * accum_steps})")

    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        model.train()
        if getattr(args, "freeze_all_backbone", False):
            keep_frozen_backbone_eval(model)
        total_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y = batch['labels'].to(args.device)

            is_last_batch = (i + 1 == len(train_loader))
            do_step = ((i + 1) % accum_steps == 0) or is_last_batch

            if amp_enabled and on_cuda:
                with autocast(dtype=amp_dtype):
                    logits, feats = model(input_ids, attention_mask, return_features="pre")
                    logits_flat = logits.reshape(-1, args.num_classes)
                    y_flat      = y.reshape(-1)
                    loss = criterion(logits_flat, y_flat)
                (loss / accum_steps).backward() if not hasattr(scaler, 'scale') else scaler.scale(loss / accum_steps).backward()
                if do_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits, feats = model(input_ids, attention_mask, return_features="pre")
                logits_flat = logits.reshape(-1, args.num_classes)
                y_flat      = y.reshape(-1)
                loss = criterion(logits_flat, y_flat)
                (loss / accum_steps).backward()
                if do_step:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

            N = len(train_loader)
        avg_train_loss = total_loss / N
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch}] Training Loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            avg_val_loss, _, _, _ = compute_validation_loss(model, val_loader, criterion, args)
            val_losses.append(avg_val_loss)
            print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")

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


    # Evaluation - load best model if validation was used
    print("\nEvaluating on test set...")
    if val_loader is not None:
        best_model_path = f"{run_folder}/best_model.pth"
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(torch.load(best_model_path, map_location=args.device))
            else:
                model.load_state_dict(torch.load(best_model_path, map_location=args.device))

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

        # Loss description
        "loss_function": args.loss_function,        # "focal" | "crossentropy"
        "loss_desc": f"supervised|{args.loss_function}",

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

    summary_df = pd.DataFrame([row])
    summary_csv_path = f"{run_folder}/run_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved run summary to {summary_csv_path}")


    print("Training completed")


if __name__ == "__main__":
    from esmfluc.training.arguments import parse_arguments
    parser = parse_arguments()
    args = parser.parse_args()
    train(args)
