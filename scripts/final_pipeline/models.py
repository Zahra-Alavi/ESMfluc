#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:09:40 2025

"""


# models.py

import torch.nn as nn
import torch
import math 

# =============================================================================
# Model Architecture
# =============================================================================

class BiLSTMClassificationModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_classes=4, dropout=0.3, bidirectional=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * (2 if self.bidirectional else 1)  # <-- important
        self.fc = nn.Linear(self.output_dim, num_classes)

    def forward(self, input_ids, attention_mask, return_features="none"):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)              # [B, L, output_dim]
        h = self.dropout(h)

        logits = self.fc(h)
        feats = h if return_features == "pre" else (logits if return_features == "post" else None)
        return logits, feats

class ESMLinearTokenClassifier(nn.Module):
    def __init__(self, embedding_model, num_classes):
        super().__init__()
        self.embedding_model = embedding_model
        self.output_dim = embedding_model.config.hidden_size
        self.fc = nn.Linear(embedding_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, return_features="none", return_attn=False):
        outputs = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attn,
            return_dict=True,
        )
        h = outputs.last_hidden_state  # [B,L,D] from ESM
        logits = self.fc(h)

        feats = None
        if return_features == "pre":
            feats = h
        elif return_features == "post":
            feats = logits

        if return_attn:
            return logits, feats, outputs.attentions
        return logits, feats


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_dim, dropout=0.0):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key   = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, return_weights=False):
        # x: [B, L, D]
        Q = self.query(x); K = self.key(x); V = self.value(x)
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)   # [B, L, L]

        if attention_mask is not None:
            # attention_mask: [B, L] with 1=valid, 0=pad
            mask = attention_mask[:, None, :].bool()                # [B, 1, L]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.bmm(attn, V)                                # [B, L, D]
        return (context, attn) if return_weights else context



class BiLSTMWithSelfAttentionModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_classes=4, dropout=0.3, bidirectional=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout
        )
        self.output_dim = hidden_size * (2 if self.bidirectional else 1)
        self.attention = SelfAttentionLayer(self.output_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.output_dim, num_classes)

    def forward(self, input_ids, attention_mask, return_attention=False, return_features="none"):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)                                      # [B, L, output_dim]
        ctx, attn = (self.attention(h, attention_mask, True) if return_attention
                     else (self.attention(h, attention_mask), None))
        ctx = self.dropout(ctx)

        logits = self.fc(ctx)
        feats = ctx if return_features == "pre" else (logits if return_features == "post" else None)
        return (logits, feats, attn) if return_attention else (logits, feats)

        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]

        # Register as buffer so it's saved inside the model state_dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x has shape [batch_size, seq_len, d_model].
        """
        seq_len = x.size(1)
        # The .to(x.device) ensures PE is on same device as input
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class TransformerClassificationModel(nn.Module):
    def __init__(self, embedding_model, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=1024, num_classes=4, dropout=0.3):
        super().__init__()
        self.embedding_model = embedding_model  # ESM or other backbone
        d_model = embedding_model.config.hidden_size

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask, return_features="none"):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        """
        # Generate embeddings from the pretrained ESM model
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Add positional encoding
        embeddings = self.pos_encoder(embeddings)

        # Create a mask for padding (True where padding is present)
        src_key_padding_mask = (attention_mask == 0)
        

        # Pass through the Transformer encoder
        # batch_first=True => shape remains [batch_size, seq_len, d_model]
        transformer_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        logits = self.fc(transformer_output)                                         # [B, L, K]

        feats = None
        if return_features == "pre":
            feats = transformer_output                                                   # pre-FC features
        elif return_features == "post":
            feats = logits                                              # post-FC features

        return logits, feats
        

# =============================================================================
# Loss Function 
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=-1):
        """
        alpha: A tensor of shape [num_classes] specifying weight for each class,
               or None if you do not want class-weighting.
        gamma: focusing parameter for Focal Loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        logpt = -self.ce_loss(inputs, targets)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        else:
            return -loss
        

