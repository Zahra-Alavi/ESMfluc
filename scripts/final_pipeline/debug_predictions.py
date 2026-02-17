#!/usr/bin/env python3
"""
Debug script to compare training evaluation vs get_attn.py predictions
"""

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import EsmModel, EsmTokenizer
import ast

from models import BiLSTMWithSelfAttentionRegressionModel, WeightedMSELoss
from data_utils import SequenceRegressionDataset, collate_fn_sequence, load_regression_data

def tokenize(sequences, tokenizer):
    return [tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False) for seq in sequences]

# Load test data
print("Loading test data...")
test_data = load_regression_data("../../data/test_data.csv")
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

X_test = tokenize(test_data['sequence'], tokenizer)
y_test = test_data['neq'].tolist()

test_dataset = SequenceRegressionDataset(X_test, y_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=lambda x: collate_fn_sequence(x, tokenizer)
)

# Load model
print("Loading model...")
device = "cpu"
embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = BiLSTMWithSelfAttentionRegressionModel(
    embedding_model=embedding_model,
    hidden_size=512,
    num_layers=2,
    num_outputs=1,
    dropout=0.3,
    bidirectional=1
)
model.to(device)

checkpoint_path = "./results/regression_bilstm_attn_unfrozen_fixed_2/best_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Loaded checkpoint from {checkpoint_path}")

# Evaluate like training does
print("\n" + "="*60)
print("EVALUATION USING DATALOADER (like training)")
print("="*60)

criterion = WeightedMSELoss(ignore_value=-100.0)
model.eval()

all_preds_loader = []
all_targets_loader = []
total_loss = 0.0
n_batches = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        
        output, feats = model(input_ids, attention_mask, return_features="pre")
        
        if output.dim() == 3 and output.size(-1) == 1:
            output = output.squeeze(-1)
        
        loss = criterion(output, y)
        total_loss += loss.item()
        n_batches += 1
        
        # Collect predictions
        output_flat = output.reshape(-1)
        y_flat = y.reshape(-1)
        mask = y_flat != -100.0
        
        all_preds_loader.extend(output_flat[mask].cpu().numpy())
        all_targets_loader.extend(y_flat[mask].cpu().numpy())

all_preds_loader = np.array(all_preds_loader)
all_targets_loader = np.array(all_targets_loader)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

mse = mean_squared_error(all_targets_loader, all_preds_loader)
mae = mean_absolute_error(all_targets_loader, all_preds_loader)
r2 = r2_score(all_targets_loader, all_preds_loader)
pearson_r, pearson_p = pearsonr(all_targets_loader, all_preds_loader)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print(f"Pearson r: {pearson_r:.4f}")
print(f"Total residues: {len(all_preds_loader)}")

# Now load predictions from JSON
print("\n" + "="*60)
print("EVALUATION USING JSON PREDICTIONS (from get_attn.py)")
print("="*60)

with open("./attn_reg_fixed2.json", 'r') as f:
    json_data = json.load(f)

# Match by sequence
all_preds_json = []
all_targets_json = []

for idx, row in test_data.iterrows():
    seq = row['sequence']
    true_neq = row['neq']
    
    # Find in JSON
    json_match = None
    for record in json_data:
        if record['sequence'] == seq:
            json_match = record
            break
    
    if json_match is None:
        print(f"ERROR: Sequence not found in JSON!")
        continue
    
    pred_neq = np.array(json_match['neq_preds'])
    
    if len(true_neq) != len(pred_neq):
        print(f"Length mismatch: true={len(true_neq)}, pred={len(pred_neq)}")
        continue
    
    all_preds_json.extend(pred_neq)
    all_targets_json.extend(true_neq)

all_preds_json = np.array(all_preds_json)
all_targets_json = np.array(all_targets_json)

mse_json = mean_squared_error(all_targets_json, all_preds_json)
mae_json = mean_absolute_error(all_targets_json, all_preds_json)
r2_json = r2_score(all_targets_json, all_preds_json)
pearson_r_json, pearson_p_json = pearsonr(all_targets_json, all_preds_json)

print(f"MSE:  {mse_json:.4f}")
print(f"RMSE: {np.sqrt(mse_json):.4f}")
print(f"MAE:  {mae_json:.4f}")
print(f"R²:   {r2_json:.4f}")
print(f"Pearson r: {pearson_r_json:.4f}")
print(f"Total residues: {len(all_preds_json)}")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"R² DataLoader: {r2:.4f}")
print(f"R² JSON:       {r2_json:.4f}")
print(f"Difference:    {abs(r2 - r2_json):.4f}")

if abs(r2 - r2_json) > 0.01:
    print("\n⚠️  WARNING: Large discrepancy detected!")
    print("Checking first sequence predictions...")
    
    # Check first sequence in detail
    first_seq = test_data['sequence'].iloc[0]
    first_true = test_data['neq'].iloc[0]
    
    # Get prediction from JSON
    for record in json_data:
        if record['sequence'] == first_seq:
            first_pred_json = np.array(record['neq_preds'])
            break
    
    # Get prediction from dataloader (single sequence)
    enc = tokenizer(first_seq, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    
    with torch.no_grad():
        output, _ = model(input_ids, attn_mask, return_features="pre")
        if output.dim() == 3 and output.size(-1) == 1:
            output = output.squeeze(-1)
        first_pred_loader = output.view(-1).cpu().numpy()
    
    print(f"\nFirst sequence comparison:")
    print(f"  Sequence length: {len(first_seq)}")
    print(f"  True Neq length: {len(first_true)}")
    print(f"  Pred from loader: {len(first_pred_loader)} values")
    print(f"  Pred from JSON: {len(first_pred_json)} values")
    print(f"  True Neq[:5]: {first_true[:5]}")
    print(f"  Loader pred[:5]: {first_pred_loader[:5]}")
    print(f"  JSON pred[:5]: {first_pred_json[:5]}")
    print(f"  Are they equal? {np.allclose(first_pred_loader, first_pred_json, atol=1e-4)}")
else:
    print("\n✓ Metrics match! Both methods give the same results.")
