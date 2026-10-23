"""Published token classifiers and their masked attention stage."""
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def train(self, mode=True):
        # Only set the inner ESM3 model to train/eval — do not touch the wrapper.
        self.esm3.train(mode)
        return self

    def eval(self):
        return self.train(False)

def run_masked_lstm(lstm, embeddings, attention_mask):
    """Run an LSTM without allowing right-padding to affect valid positions."""
    if attention_mask is None:
        return lstm(embeddings)

    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("Every sequence must contain at least one unmasked token.")

    packed = pack_padded_sequence(
        embeddings,
        lengths.detach().cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, hidden = lstm(packed)
    output, _ = pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=embeddings.size(1),
    )
    return output, hidden

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
                 num_classes=2, dropout=0.3, bidirectional=1):
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
        h, _ = run_masked_lstm(self.lstm, emb, attention_mask)     # [B, L, output_dim]
        ctx, attn = (self.attention(h, attention_mask, True) if return_attention
                     else (self.attention(h, attention_mask), None))
        ctx = self.dropout(ctx)

        logits = self.fc(ctx)
        feats = ctx if return_features == "pre" else (logits if return_features == "post" else None)
        return (logits, feats, attn) if return_attention else (logits, feats)

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
