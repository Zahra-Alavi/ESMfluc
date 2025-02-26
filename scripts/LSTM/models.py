#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:09:40 2025

@author: zalavi
"""


# models.py

import torch.nn as nn
import torch

# =============================================================================
# Model Architecture
# =============================================================================

class BiLSTMClassificationModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers, num_classes=4, dropout=0.3):
        super().__init__()
        self.embedding_model = embedding_model
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(embeddings)     # [batch_size, seq_len, hidden_size*2]
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)              # [batch_size, seq_len, num_classes]
        return logits

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.key = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.value = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_out):
        Q = self.query(lstm_out)
        K = self.key(lstm_out)
        V = self.value(lstm_out)

        # [batch_size, seq_len, seq_len]
        attention_scores = torch.bmm(Q, K.transpose(1, 2))
        attention_weights = self.softmax(attention_scores)
        context = torch.bmm(attention_weights, V)  # [batch_size, seq_len, hidden_size*2]
        return context

class BiLSTMWithSelfAttentionModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers, num_classes=4, dropout=0.3):
        super().__init__()
        self.embedding_model = embedding_model
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = SelfAttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(embeddings)     # [batch_size, seq_len, hidden_size*2]
        context = self.attention(lstm_out)
        context = self.dropout(context)
        logits = self.fc(context)
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
        