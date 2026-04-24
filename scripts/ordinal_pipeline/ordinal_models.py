import torch
import torch.nn as nn


class OrdinalBiLSTMModel(nn.Module):
    def __init__(
        self,
        embedding_model,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        num_thresholds=3,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)

        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )

        self.output_dim = hidden_size * (2 if self.bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.output_dim, num_thresholds)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        h, _ = self.lstm(emb)
        h = self.dropout(h)
        logits = self.fc(h)
        return logits
