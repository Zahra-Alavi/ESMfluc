#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:09:40 2025

@author: zalavi
"""


# models.py

import torch.nn as nn
import torch
import math 

# =============================================================================
# Model Architecture
# =============================================================================

class LSTMClassificationModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers, num_classes=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding_model = embedding_model
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(embeddings)     # [batch_size, seq_len, hidden_size*2]
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)              # [batch_size, seq_len, num_classes]
        return logits

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, bidirectional=True):
        super().__init__()
        if bidirectional:
            hidden_size = hidden_size * 2
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_out, return_weights=False):
        Q = self.query(lstm_out)                        # [batch_size, seq_len, hidden*2]
        K = self.key(lstm_out)                          # same shape
        V = self.value(lstm_out)                        # same shape

        attention_scores = torch.bmm(Q, K.transpose(1, 2))   # [batch_size, seq_len, seq_len]
        attention_weights = self.softmax(attention_scores)   # softmax over dim=-1
        context = torch.bmm(attention_weights, V)            # [batch_size, seq_len, hidden*2]

        if return_weights:
            # Return both the context and the raw attention weights
            return context, attention_weights
        else:
            return context
        
class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_heads=8, bidirectional=False):
        super().__init__()
        self.input_size = hidden_size * 2 if bidirectional else hidden_size
        assert self.input_size % n_heads == 0, "Input size must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = self.input_size // n_heads
        self.bidirectional = bidirectional

        self.query = nn.Linear(self.input_size, hidden_size)
        self.key = nn.Linear(self.input_size, hidden_size)
        self.value = nn.Linear(self.input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, self.input_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, lstm_out, return_weights = False):
        batch_size, seq_len, hidden_size = lstm_out.size()
        
        Q = self.query(lstm_out)
        K = self.key(lstm_out)
        V = self.value(lstm_out)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attention = self.softmax(energy)
        context = torch.matmul(attention, V)
        
        output_dim = hidden_size//2 if self.bidirectional else hidden_size
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, output_dim)
        context = self.fc_out(context)
        
        if return_weights:
            return context, attention
        else:
            return context
        
        
class LSTMWithMultiHeadAttentionModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers, num_heads=8, num_classes=4, dropout=0.3, bidirectional=False):
        super().__init__()
        self.embedding_model = embedding_model
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.multihead_attention = MultiHeadSelfAttentionLayer(hidden_size, num_heads, bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, input_ids, attention_mask, return_attention=False):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(embeddings)     # [batch_size, seq_len, hidden_size*2]
        
        if return_attention:
            context, attn_weights = self.multihead_attention(lstm_out)
        else:
            context = self.multihead_attention(lstm_out)

        context = self.dropout(context)

        logits = self.fc(context)  # [batch_size, seq_len, num_classes]

        if return_attention:
            return logits, attn_weights
        else:
            return logits


class LSTMWithSelfAttentionModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers, num_classes=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding_model = embedding_model
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.attention = SelfAttentionLayer(hidden_size, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, return_attention=False):
        # 1) Obtain embeddings from your pretrained ESM
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 2) Pass embeddings to LSTM
        lstm_out, _ = self.lstm(embeddings)     # [batch_size, seq_len, hidden_size*2]

        # 3) Pass LSTM output to your custom self-attention
        if return_attention:
            context, attn_weights = self.attention(lstm_out, return_weights=True)
        else:
            context = self.attention(lstm_out)

        context = self.dropout(context)

        # 4) Final classification layer
        logits = self.fc(context)  # [batch_size, seq_len, num_classes]

        if return_attention:
            return logits, attn_weights
        else:
            return logits
        
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

    def forward(self, input_ids, attention_mask):
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

        # Final classification layer
        logits = self.fc(transformer_output)   # [batch_size, seq_len, num_classes]
        return logits
        

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
        

