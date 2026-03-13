import os
import torch
import torch.nn as nn
from pathlib import Path
from transformers import EsmModel
import huggingface_hub

from esm.pretrained import ESM3_sm_open_v0

# Load HF_TOKEN from .env if present and not already set
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    huggingface_hub.login(token=_hf_token, add_to_git_credential=False)

class EsmFlucModel(nn.Module):
    def __init__(self, pretrained_model_name='facebook/esm2_t6_8M_UR50D', 
                 hidden_size=512, num_unfreeze_layers=0, dropout_rate=0.1, 
                 use_temperature=True):
        super().__init__()
        
        self.is_esm3 = "esm3" in pretrained_model_name.lower()
        self.use_temperature = use_temperature

        if self.is_esm3:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            self.esm = ESM3_sm_open_v0(device)
            esm_hidden_size = 1536 
            
            for param in self.esm.parameters():
                param.requires_grad = False

            # ESM3 uses 'transformer.blocks' for its layers
            if num_unfreeze_layers > 0:
                for block in self.esm.transformer.blocks[-num_unfreeze_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
        else:
            # ESM2 Logic (HuggingFace)
            self.esm = EsmModel.from_pretrained(pretrained_model_name)
            esm_hidden_size = self.esm.config.hidden_size
            
            for param in self.esm.parameters():
                param.requires_grad = False
            
            if num_unfreeze_layers > 0:
                for layer in self.esm.encoder.layer[-num_unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        input_size = esm_hidden_size + 1 if use_temperature else esm_hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input_ids=None, attention_mask=None, temperature=None, token_ids=None):
        if self.is_esm3:
            # ESM3.forward() returns ESMOutput; .embeddings is the per-residue
            # hidden state tensor [batch, seq_len, 1536] before the output heads.
            output = self.esm(sequence_tokens=token_ids)
            last_hidden_state = output.embeddings  # [B, L, 1536]
        else:
            # ESM2 logic
            embeddings = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = embeddings.last_hidden_state 

        # Temperature concatenation
        if self.use_temperature:
            batch_size, seq_len, _ = last_hidden_state.shape
            temperature_expanded = temperature.view(batch_size, 1, 1).expand(-1, seq_len, -1)
            last_hidden_state = torch.cat((last_hidden_state, temperature_expanded), dim=-1)

        return self.regression_head(last_hidden_state).squeeze(-1)