#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:11:16 2025

@author: zalavi
"""

# attention_visualizer.py
import torch
import matplotlib.pyplot as plt
from transformers import EsmModel, EsmTokenizer

from models import BiLSTMWithSelfAttentionModel

device = "cuda" if torch.cuda.is_available() else "cpu"


embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

embedding_model.to(device)



freeze_list = range(0, 4+1)
for name, param in embedding_model.named_parameters():
    if "encoder.layer" in name:
        layer_num = int(name.split(".")[2])
        param.requires_grad = not layer_num in freeze_list
    else:   
        param.requires_grad = True
        
        

model = BiLSTMWithSelfAttentionModel(
    embedding_model=embedding_model,
    hidden_size=512,
    num_layers=3,
    num_classes=2,
    dropout=0.3
)
model.to(device)


checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint)


tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")


def visualize_attention(model, tokenizer, seq):
    enc = tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(input_ids, attn_mask, return_attention=True)

    # attn_weights: [batch_size, seq_len, seq_len]
    attn_weights = attn_weights[0].cpu().numpy()    # shape = [seq_len, seq_len]
    
    token_ids = input_ids[0].tolist()  # shape = [seq_len], since batch_size=1
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
  
    plt.figure(dpi=500)
    plt.imshow(attn_weights, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Attention Heatmap")
    seq_len = attn_weights.shape[0]

   
    step = 25
    indices = range(0, seq_len, step)

    # Build custom labels that show "resNum-aminoAcid"
    xtick_labels = [f"{i+1}-{tokens[i]}" for i in indices]

    plt.xticks(indices, xtick_labels, rotation=90, fontsize=12)
    plt.yticks(indices, xtick_labels, fontsize=12)

    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.savefig("test_set_row12.png", dpi=500)
    plt.show()
  

if __name__ == "__main__":
    test_sequence = "VSGSSSVGEMSGRSVSQQTSDQYANNLAGRTESPQGSSLASRIIERLSSVAHSVIGFIQRMFSEGSHKP"
    visualize_attention(model, tokenizer, test_sequence)
