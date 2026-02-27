import torch
import torch.nn as nn
from transformers import EsmModel
from esm.pretrained import ESM3_sm_open_v0

class EsmFlucModel(nn.Module):
    def __init__(self, pretrained_model_name='facebook/esm2_t6_8M_UR50D', 
                 hidden_size=512, num_unfreeze_layers=0, dropout_rate=0.1, 
                 use_temperature=True):
        super().__init__()
        
        self.is_esm3 = "esm3" in pretrained_model_name.lower()
        self.use_temperature = use_temperature

        if self.is_esm3:
            # Dimensions for esm3-sm-open-v0 is 1536
            self.esm = ESM3_sm_open_v0()
            esm_hidden_size = 1536 
            
            for param in self.esm.parameters():
                param.requires_grad = False

            if num_unfreeze_layers > 0:
                for block in self.esm.transformer.blocks[-num_unfreeze_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
        else:

            self.esm = EsmModel.from_pretrained(pretrained_model_name)
            esm_hidden_size = self.esm.config.hidden_size
            
            for param in self.esm.parameters():
                param.requires_grad = False
            
            if num_unfreeze_layers > 0:
                for layer in self.esm.encoder.layer[-num_unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # Regression head automatically adjusts to backbone size
        input_size = esm_hidden_size + 1 if use_temperature else esm_hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input_ids=None, attention_mask=None, temperature=None, token_ids=None):
        if self.is_esm3:
            # ESM3 forward pass specifically extracts residue embeddings
            # token_ids is the sequence track from your Esm3SequenceDataset
            output = self.esm.forward_track(sequence_tokens=token_ids)
            last_hidden_state = output.sequence
        else:
            # ESM2 forward pass
            embeddings = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = embeddings.last_hidden_state 

        # Temperature concatenation
        if self.use_temperature:
            batch_size, seq_len, _ = last_hidden_state.shape
            temperature_expanded = temperature.view(batch_size, 1, 1).expand(-1, seq_len, -1)
            last_hidden_state = torch.cat((last_hidden_state, temperature_expanded), dim=-1)

        return self.regression_head(last_hidden_state).squeeze(-1)