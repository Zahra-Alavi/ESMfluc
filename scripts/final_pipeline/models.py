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
# ESM3 Compatibility Wrapper
# =============================================================================

class _ESM3Output:
    """Minimal stand-in for HuggingFace model output with .last_hidden_state."""
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _DummyConfig:
    """Provides .hidden_size so model __init__ can read embedding dim."""
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class ESM3Wrapper(nn.Module):
    """
    Wraps ESM3_sm_open_v0 to expose the same interface as HuggingFace EsmModel:
      - .config.hidden_size  (1536 for esm3_sm_open_v1)
      - forward(input_ids, attention_mask) -> object with .last_hidden_state

    ESM3's raw forward uses 'sequence_tokens' and returns an object whose
    per-residue representations may live in different fields across library
    versions; this wrapper normalises all of them to .last_hidden_state.
    """
    ESM3_HIDDEN = 1536  # esm3_sm_open_v1 hidden dimension

    def __init__(self, esm3_model):
        super().__init__()
        self.esm3 = esm3_model
        self.config = _DummyConfig(self.ESM3_HIDDEN)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # kwargs absorbs HF-style arguments the calling code may pass
        out = self.esm3(sequence_tokens=input_ids)
        if hasattr(out, 'last_hidden_state'):
            h = out.last_hidden_state
        elif hasattr(out, 'sequence_last_hidden_states'):
            h = out.sequence_last_hidden_states
        elif hasattr(out, 'embeddings'):
            h = out.embeddings
        else:
            raise AttributeError(
                f"Cannot find hidden states in ESM3 output. "
                f"Available fields: {list(vars(out).keys())}")
        return _ESM3Output(h)

    def parameters(self, recurse=True):
        return self.esm3.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.esm3.named_parameters(prefix, recurse)


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


# =============================================================================
# Regression Models
# =============================================================================

class BiLSTMRegressionModel(nn.Module):
    """BiLSTM for per-residue regression (e.g., predicting continuous Neq values)"""
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_outputs=1, dropout=0.3, bidirectional=1, activation='none'):
        super().__init__()
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.activation = activation  # 'none', 'bounded_sigmoid', 'exp'
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(self.output_dim, num_outputs)

    def forward(self, input_ids, attention_mask, return_features="none"):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)
        h = self.dropout(h)
        output = self.fc(h)  # [B, L, num_outputs]
        
        # Apply activation if specified
        if self.activation == 'bounded_sigmoid':
            # For Neq ∈ [1.0, 8.0] - use WITHOUT log transform
            output = 1.0 + 7.0 * torch.sigmoid(output)
        elif self.activation == 'log_bounded_sigmoid':
            # For log(Neq) ∈ [0.0, 2.23] - use WITH log transform
            output = 2.23 * torch.sigmoid(output)
        # else: 'none' - raw output
        
        feats = h if return_features == "pre" else (output if return_features == "post" else None)
        return output, feats


class BiLSTMWithSelfAttentionRegressionModel(nn.Module):
    """BiLSTM with self-attention for regression"""
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_outputs=1, dropout=0.3, bidirectional=1, activation='none'):
        super().__init__()
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.activation = activation  # 'none', 'bounded_sigmoid', 'exp'
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
        self.fc = nn.Linear(self.output_dim, num_outputs)

    def forward(self, input_ids, attention_mask, return_attention=False, return_features="none"):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)
        ctx, attn = (self.attention(h, attention_mask, True) if return_attention
                     else (self.attention(h, attention_mask), None))
        ctx = self.dropout(ctx)
        output = self.fc(ctx)  # [B, L, num_outputs]
        
        # Apply activation if specified
        if self.activation == 'bounded_sigmoid':
            # For Neq ∈ [1.0, 8.0] - use WITHOUT log transform
            output = 1.0 + 7.0 * torch.sigmoid(output)
        elif self.activation == 'log_bounded_sigmoid':
            # For log(Neq) ∈ [0.0, 2.23] - use WITH log transform
            output = 2.23 * torch.sigmoid(output)
        # else: 'none' - raw output
        
        feats = ctx if return_features == "pre" else (output if return_features == "post" else None)
        return (output, feats, attn) if return_attention else (output, feats)


class TransformerRegressionModel(nn.Module):
    """Transformer for regression"""
    def __init__(self, embedding_model, nhead=8, num_encoder_layers=6,
                 dim_feedforward=1024, num_outputs=1, dropout=0.3):
        super().__init__()
        self.embedding_model = embedding_model
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
        self.fc = nn.Linear(d_model, num_outputs)

    def forward(self, input_ids, attention_mask, return_features="none"):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        embeddings = self.pos_encoder(embeddings)
        src_key_padding_mask = (attention_mask == 0)
        
        transformer_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.fc(transformer_output)  # [B, L, num_outputs]

        feats = transformer_output if return_features == "pre" else (output if return_features == "post" else None)
        return output, feats


class ESMLinearTokenRegressor(nn.Module):
    """ESM embeddings + linear layer for regression"""
    def __init__(self, embedding_model, num_outputs=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.output_dim = embedding_model.config.hidden_size
        self.fc = nn.Linear(embedding_model.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask, return_features="none", return_attn=False):
        outputs = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attn,
            return_dict=True,
        )
        h = outputs.last_hidden_state
        output = self.fc(h)  # [B, L, num_outputs]

        feats = h if return_features == "pre" else (output if return_features == "post" else None)
        
        if return_attn:
            return output, feats, outputs.attentions
        return output, feats


# =============================================================================
# Regression Loss Functions
# =============================================================================

class WeightedMSELoss(nn.Module):
    """MSE loss with optional per-residue weighting and padding mask"""
    def __init__(self, reduction='mean', ignore_value=-100.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_value = ignore_value
    
    def forward(self, predictions, targets, weights=None):
        """
        predictions: [B, L, num_outputs] or [B, L]
        targets: [B, L, num_outputs] or [B, L]
        weights: optional [B, L] weighting mask
        """
        # Ensure same shape
        if targets.dim() == 2 and predictions.dim() == 3:
            targets = targets.unsqueeze(-1)
        if predictions.dim() == 2 and targets.dim() == 3:
            predictions = predictions.unsqueeze(-1)
        
        # Mask out padding (where target == ignore_value)
        mask = (targets != self.ignore_value).float()
        
        # Compute squared error
        sq_error = (predictions - targets) ** 2
        
        # Apply mask
        sq_error = sq_error * mask
        
        # Apply optional weights
        if weights is not None:
            if weights.dim() == 2 and sq_error.dim() == 3:
                weights = weights.unsqueeze(-1)
            sq_error = sq_error * weights
        
        if self.reduction == 'mean':
            return sq_error.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return sq_error.sum()
        else:
            return sq_error


# =============================================================================
# Ordinal Regression Models and Loss (CORAL-style)
# =============================================================================

class OrdinalCrossEntropyLoss(nn.Module):
    """
    CORAL-style ordinal loss for labels in {0, ..., K-1}.

    Model output is expected to be logits with shape [B, L, K-1] or [N, K-1],
    where each logit predicts P(y > k) for threshold k in [0, K-2].
    """
    def __init__(self, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # logits: [B, L, K-1] or [N, K-1]
        # targets: [B, L] or [N] with integer ordinal labels
        if logits.dim() == 3:
            n_classes_minus_1 = logits.size(-1)
            logits_flat = logits.reshape(-1, n_classes_minus_1)
            targets_flat = targets.reshape(-1)
        else:
            n_classes_minus_1 = logits.size(-1)
            logits_flat = logits
            targets_flat = targets

        valid_mask = (targets_flat != self.ignore_index)
        if valid_mask.sum() == 0:
            # Keep gradient graph valid
            return logits_flat.sum() * 0.0

        logits_valid = logits_flat[valid_mask]                     # [M, K-1]
        targets_valid = targets_flat[valid_mask].long()            # [M]

        # Build cumulative binary targets: t_k = 1 if y > k else 0
        levels = torch.arange(n_classes_minus_1, device=logits_valid.device).unsqueeze(0)
        ordinal_targets = (targets_valid.unsqueeze(1) > levels).float()  # [M, K-1]

        loss_matrix = self.bce(logits_valid, ordinal_targets)
        if self.reduction == 'sum':
            return loss_matrix.sum()
        if self.reduction == 'none':
            return loss_matrix
        return loss_matrix.mean()


def ordinal_logits_to_probs(logits):
    """
    Convert ordinal logits [B, L, K-1] into class probabilities [B, L, K].
    Uses cumulative-link identity with p_k = P(y=k).
    """
    q = torch.sigmoid(logits)  # q_k = P(y > k)
    left = torch.ones(*q.shape[:-1], 1, device=q.device, dtype=q.dtype)
    right = torch.zeros(*q.shape[:-1], 1, device=q.device, dtype=q.dtype)
    q_ext = torch.cat([left, q, right], dim=-1)
    probs = q_ext[..., :-1] - q_ext[..., 1:]
    return probs.clamp_min(0.0)


class BiLSTMOrdinalRegressionModel(nn.Module):
    """BiLSTM ordinal model with CORAL-style thresholds."""
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_classes=4, dropout=0.3, bidirectional=1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal regression")
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(self.output_dim, num_classes - 1)

    def forward(self, input_ids, attention_mask, return_features="none", return_probs=False):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)
        h = self.dropout(h)
        logits = self.fc(h)  # [B, L, K-1]

        feats = h if return_features == "pre" else (logits if return_features == "post" else None)
        if return_probs:
            probs = ordinal_logits_to_probs(logits)
            return logits, feats, probs
        return logits, feats


class BiLSTMWithSelfAttentionOrdinalRegressionModel(nn.Module):
    """BiLSTM + SelfAttention ordinal model with CORAL-style thresholds."""
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_classes=4, dropout=0.3, bidirectional=1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal regression")
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.num_classes = num_classes
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
        self.fc = nn.Linear(self.output_dim, num_classes - 1)

    def forward(self, input_ids, attention_mask, return_attention=False, return_features="none", return_probs=False):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)
        ctx, attn = (self.attention(h, attention_mask, True) if return_attention
                     else (self.attention(h, attention_mask), None))
        ctx = self.dropout(ctx)
        logits = self.fc(ctx)  # [B, L, K-1]

        feats = ctx if return_features == "pre" else (logits if return_features == "post" else None)
        if return_probs:
            probs = ordinal_logits_to_probs(logits)
            if return_attention:
                return logits, feats, attn, probs
            return logits, feats, probs
        return (logits, feats, attn) if return_attention else (logits, feats)


class TransformerOrdinalRegressionModel(nn.Module):
    """Transformer ordinal model with CORAL-style thresholds."""
    def __init__(self, embedding_model, nhead=8, num_encoder_layers=6,
                 dim_feedforward=1024, num_classes=4, dropout=0.3):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal regression")
        self.embedding_model = embedding_model
        self.num_classes = num_classes
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
        self.fc = nn.Linear(d_model, num_classes - 1)

    def forward(self, input_ids, attention_mask, return_features="none", return_probs=False):
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        embeddings = self.pos_encoder(embeddings)
        src_key_padding_mask = (attention_mask == 0)

        h = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        logits = self.fc(h)  # [B, L, K-1]

        feats = h if return_features == "pre" else (logits if return_features == "post" else None)
        if return_probs:
            probs = ordinal_logits_to_probs(logits)
            return logits, feats, probs
        return logits, feats


class ESMLinearTokenOrdinalRegressor(nn.Module):
    """ESM embeddings + linear ordinal head (CORAL-style thresholds)."""
    def __init__(self, embedding_model, num_classes=4):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal regression")
        self.embedding_model = embedding_model
        self.output_dim = embedding_model.config.hidden_size
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_model.config.hidden_size, num_classes - 1)

    def forward(self, input_ids, attention_mask, return_features="none", return_attn=False, return_probs=False):
        outputs = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attn,
            return_dict=True,
        )
        h = outputs.last_hidden_state
        logits = self.fc(h)  # [B, L, K-1]

        feats = h if return_features == "pre" else (logits if return_features == "post" else None)

        if return_probs:
            probs = ordinal_logits_to_probs(logits)
            if return_attn:
                return logits, feats, outputs.attentions, probs
            return logits, feats, probs

        if return_attn:
            return logits, feats, outputs.attentions
        return logits, feats
