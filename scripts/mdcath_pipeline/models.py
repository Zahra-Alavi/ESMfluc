import torch
import torch.nn as nn
from transformers import EsmModel

class EsmFlucModel(nn.Module):
    def __init__(self, pretrained_model_name='facebook/essm2_t6_8M_UR50D', hidden_size=180):
        super().__init__()

        self.esm = EsmModel.from_pretrained(pretrained_model_name)
        
        # Regression head - ESM last hidden state + 1 temperature feature (scalar)
        # hidden size for t6 is 320, for t33 is 1280
        esm_hidden_size = self.esm.config.hidden_size
        
        self.regression_head = nn.Sequential(
            nn.Linear(esm_hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output is a single scalar per residue
        )
    
    def forward(self, input_ids, attention_mask, temperature):
        # Get residue-level embeddings from ESM
        embeddings = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = embeddings.last_hidden_state # [batch, seq_len, d_model]

        # Expand temperature [batch, 1] to [batch, seq_len, 1] to match ESM output
        batch_size, seq_len, _ = last_hidden_state.shape
        temperature_expanded = temperature.view(batch_size, 1, 1).expand(-1, seq_len, -1)
        combined = torch.cat((last_hidden_state, temperature_expanded), dim=-1)

        predictions = self.regression_head(combined).squeeze(-1) # [batch, seq_len]
        return predictions