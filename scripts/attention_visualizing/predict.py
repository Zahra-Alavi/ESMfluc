#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:55:41 2025

@author: zalavi
"""


"""
This script takes the test set in csv format
and returns a json file that includes:
    the test set along with predicted neq class using the best model checkpoint. 
    output: test_data_with_predictions.json
"""
import pandas as pd
import ast
import torch 
from transformers import EsmModel, EsmTokenizer

from models import BiLSTMWithSelfAttentionModel

device = "cuda" if torch.cuda.is_available() else "cpu"


input_file = "../../data/test_data.csv"
test_data = pd.read_csv(input_file)

# classify neq values to get GT label
test_data['neq'] = test_data['neq'].apply(ast.literal_eval)
test_data['neq_class'] = test_data['neq'].apply(lambda x:[0 if float(val)==1.0 else 1 for val in x])


def check_lengths(row):
    return len(row['sequence']) == len(row['neq']) == len(row['neq_class']) 

length_ok = test_data.apply(check_lengths, axis=1)
invalid_rows = test_data[~length_ok]
print(f"Number of rows with length mismatch in the initial test set: {len(invalid_rows)}")


def run_model(model, tokenizer, sequence, device):     
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        logits, attn_weights_torch = model(input_ids, attn_mask, return_attention=True)
        # attn_weights_torch shape: [batch_size, seq_len, seq_len]
    attn_weights = attn_weights_torch[0].cpu().numpy()  # shape=(L,L)
    
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # length=L
    
    y_probs = torch.softmax(logits, dim=-1)
    y_preds = torch.argmax(y_probs, dim=-1)
    y_preds = y_preds.view(-1)
    
    return attn_weights, tokens, y_preds


embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
embedding_model.to(device)


model = BiLSTMWithSelfAttentionModel(
    embedding_model=embedding_model,
    hidden_size=512,
    num_layers=3,
    num_classes=2,
    dropout=0.3
)
model.to(device)


checkpoint = torch.load('../../results/best_model.pth', map_location=device)
model.load_state_dict(checkpoint)

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

model.eval()

all_preds = []

for i, row in test_data.iterrows():
    sequence = row['sequence']
    print(f'sequence length: {len(sequence)}')
    
    try:
        _, _, y_preds = run_model(model, tokenizer, sequence, device)
        preds_list = y_preds.cpu().tolist()
        print(f'predicted label length: {len(preds_list)}')
        all_preds.append(preds_list)  
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        all_preds.append([None]*len(sequence))  

test_data['pred_class'] = all_preds

def check_lengths(row):
    return len(row['sequence']) == len(row['neq']) == len(row['neq_class']) == len(row['pred_class'])

length_ok = test_data.apply(check_lengths, axis=1)
invalid_rows = test_data[~length_ok]
print(f"Number of rows with length mismatch in the test set after inference: {len(invalid_rows)}")


test_data.to_json("../../results/test_data_with_predictions.json", orient="records")

