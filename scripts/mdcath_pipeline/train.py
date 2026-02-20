import argparse
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer

from models import EsmFlucModel
from dataset import MdCathSequenceDataset
from trainer_module import EsmFlucTrainer

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="ESM-Flex: Protein Flexibility Prediction")

    parser.add_argument("--test-only", action="store_true", help="Run validation only without training")
    # --- Data & Path Arguments ---
    data_group = parser.add_argument_group("Data & Paths")
    data_group.add_argument("--train_path", type=str, default="../../data/mdcath/train_split_mmseqs2.csv")
    data_group.add_argument("--val_path", type=str, default="../../data/mdcath/test_split_mmseqs2.csv")
    data_group.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    data_group.add_argument("--checkpoint_path", type=str, default=None, help="Path to a specific .ckpt file for testing/generalization.")
    data_group.add_argument("--temperatures", type=str, default="320,348,379,413,450", help="Comma-separated list of temperatures to include as input features. Default: '320,348,379,413,450'")

    # --- Model Architecture Arguments ---
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    model_group.add_argument("--hidden_size", type=int, default=512, help="Regressor head hidden dimension")
    model_group.add_argument("--max_len", type=int, default=1024)
    model_group.add_argument("--masked_value", type=int, default=-100)
    model_group.add_argument("--num_unfreeze_layers", type=int, default=0, help="Number of ESM layers to unfreeze for fine-tuning (starting from the last layer). 0 means all layers are frozen.")
    model_group.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the regression head")

    # --- Training Hyperparameters ---
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--batch_size", type=int, default=16)
    train_group.add_argument("--lr", type=float, default=1e-5, help="Base LR (auto-scaled by num_gpus)")
    train_group.add_argument("--weight_decay", type=float, default=1e-2)
    train_group.add_argument("--epochs", type=int, default=20)
    train_group.add_argument("--early_stop_patience", type=int, default=3)
    train_group.add_argument("--seed", type=int, default=42)

    # --- Loss Function Configuration ---
    loss_group = parser.add_argument_group("Loss Function")
    loss_group.add_argument("--loss_type", type=str, choices=['weighted', 'standard'], default='weighted')
    loss_group.add_argument("--weight_threshold", type=float, default=3.0)
    loss_group.add_argument("--weight_factor", type=float, default=5.0)

    # --- Computational Resources ---
    comp_group = parser.add_argument_group("Computational Resources")
    comp_group.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    L.seed_everything(args.seed)

    # --- Hardware & Strategy Detection ---
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_gpus > 0:
        accelerator, devices, strategy, precision = "gpu", num_gpus, ("ddp_find_unused_parameters_true" if num_gpus > 1 else "auto"), "16-mixed"
        print(f"Training on {num_gpus} GPU(s) using {strategy} strategy.")
    elif torch.backends.mps.is_available():
        accelerator, devices, strategy, precision = "mps", 1, "auto", "32-true"
        print("Training on Apple Silicon (MPS).")
    else:
        accelerator, devices, strategy, precision = "cpu", "auto", "auto", "32-true"
        print("No GPU found. Training on CPU.")

    # --- Initialize Tokenizer & Data ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    
    # Select only the specified temperature columns and drop others
    temperature_list = [int(t) for t in args.temperatures.split(",")]
    
    # Need to handle for different dataset formats - mdcath has multiple temperatures as separate columns, atlas has a single 'neq' column without temperature suffix
    temp_cols = []
    if all(f"neq_{temp}" in train_df.columns for temp in temperature_list):
        temp_cols = [f"neq_{temp}" for temp in temperature_list]
    elif "neq" in train_df.columns:
        temp_cols = ["neq"]
    else:
        raise ValueError("No valid temperature columns found in the dataset. Please check the column names and the --temperatures argument.")
    
    target_cols = temp_cols + ['sequence']
    train_df = train_df[target_cols]
    val_df = val_df[target_cols]

    # Determine dynamic max length
    data_max_len = max(train_df['sequence'].str.len().max(), val_df['sequence'].str.len().max()) + 2
    final_max_len = min(data_max_len, args.max_len)
    print(f"Max sequence length: {data_max_len}. Using max_len={final_max_len}.")
    
    train_dataset = MdCathSequenceDataset(train_df, tokenizer=tokenizer, max_length=final_max_len, masked_value=args.masked_value)
    val_dataset = MdCathSequenceDataset(val_df, tokenizer=tokenizer, max_length=final_max_len, masked_value=args.masked_value)

    pin_memory = True if num_gpus > 0 else False
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=True
    )

    # --- Initialize Model and Trainer ---
    # Scaled Learning Rate
    effective_lr = args.lr * max(1, num_gpus)
    
    model = EsmFlucModel(pretrained_model_name=args.model_name, hidden_size=args.hidden_size, num_unfreeze_layers=args.num_unfreeze_layers, dropout_rate=args.dropout_rate, use_temperature=len(temp_cols) > 1)
    trainer_module = EsmFlucTrainer(model, lr=effective_lr, weight_threshold=args.weight_threshold, weight_factor=args.weight_factor, weight_decay=args.weight_decay, masked_value=args.masked_value, loss_type=args.loss_type)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=args.checkpoint_dir,
        filename="esm-fluc-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3, mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=args.early_stop_patience,
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=precision,
        use_distributed_sampler=(num_gpus > 1) 
    )
    
        
    if args.test_only:
        if args.checkpoint_path is None:
            raise ValueError("You must provide --checkpoint_path for test_only mode.")
        print(f"--- Running Generalization Test on {args.temperatures}K ---")
        trainer.test(trainer_module, dataloaders=val_loader, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(trainer_module, train_loader, val_loader)

if __name__ == "__main__":
    main()