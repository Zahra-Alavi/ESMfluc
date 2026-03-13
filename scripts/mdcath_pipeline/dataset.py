import torch
import ast
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

# ── Compatibility patch ────────────────────────────────────────────────────────
# EsmSequenceTokenizer (ESM 3.x) was written for older transformers where
# PreTrainedTokenizerFast exposed special tokens via __getattr__.
# Transformers ≥ 4.40 removed that __getattr__, causing two failures:
#   1. __init__: base class calls setattr(self, 'pad_token', ...) but those
#      are read-only properties → AttributeError.
#   2. _get_token: calls self.__getattr__(name) which no longer exists.
# Fix: add no-op setters + replace _get_token with a direct string lookup.
_ESM3_SPECIAL_TOKENS = {
    "cls_token": "<cls>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "eos_token": "<eos>",
    "bos_token": "<cls>",
    "unk_token": "<unk>",
}

def _esm_get_token(self, token_name: str) -> str:
    if token_name in _ESM3_SPECIAL_TOKENS:
        return _ESM3_SPECIAL_TOKENS[token_name]
    raise AttributeError(f"Unknown special token name: {token_name!r}")

def _noop_setter(self, value):
    pass

for _attr in ("cls_token", "eos_token", "mask_token", "pad_token", "bos_token"):
    _prop = getattr(EsmSequenceTokenizer, _attr, None)
    if isinstance(_prop, property) and _prop.fset is None:
        setattr(EsmSequenceTokenizer, _attr, _prop.setter(_noop_setter))

EsmSequenceTokenizer._get_token = _esm_get_token
# ──────────────────────────────────────────────────────────────────────────────

class BaseSequenceDataset(Dataset):
    """Shared logic for all protein flexibility datasets."""
    def __init__(self, dataframe, max_length=1024, masked_value=-100, use_log_scaling=False):
        self.df = dataframe
        self.max_len = max_length
        self.masked_value = masked_value
        self.use_log_scaling = use_log_scaling
        
        # Determine Temperature Columns
        self.temp_cols = [c for c in dataframe.columns if c.startswith('neq')]
        if self.temp_cols:
            if self.temp_cols[0].startswith('neq_'):
                self.temperatures = [float(col.split('_')[1]) for col in self.temp_cols]
            else:
                self.temperatures = [300.0]
        self.max_temp = max(self.temperatures)
        self.min_temp = min(self.temperatures)

    def __len__(self):
        return len(self.df) * len(self.temp_cols)

    def _get_shared_data(self, idx):
        """Common logic to get sequence, normalized temp, and processed labels."""
        domain_idx = idx // len(self.temp_cols)
        temp_idx = idx % len(self.temp_cols)
        row = self.df.iloc[domain_idx]
        
        # Temperature Normalization
        temp = self.temperatures[temp_idx]
        if len(self.temperatures) > 1:
            temp = (float(temp) - self.min_temp) / (self.max_temp - self.min_temp)
        
        # Label Processing
        labels = ast.literal_eval(row[self.temp_cols[temp_idx]])
        labels = torch.tensor(labels, dtype=torch.float)
        if self.use_log_scaling:
            labels = torch.log1p(labels)
            
        # Padded Label Tensor
        target_neq = torch.full((self.max_len,), self.masked_value, dtype=torch.float32)
        actual_len = min(len(labels), self.max_len - 2)
        target_neq[1:actual_len+1] = labels[:actual_len]
        
        return row['sequence'], torch.tensor(temp, dtype=torch.float), target_neq

class MdCathSequenceDataset(BaseSequenceDataset):
    def __init__(self, dataframe, tokenizer, **kwargs):
        super().__init__(dataframe, **kwargs)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sequence, temp, target_neq = self._get_shared_data(idx)
        
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_neq,
            'temperature': temp
        }

class Esm3SequenceDataset(BaseSequenceDataset):
    def __init__(self, dataframe, tokenizer, **kwargs):
        super().__init__(dataframe, **kwargs)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sequence, temp, target_neq = self._get_shared_data(idx)

        # EsmSequenceTokenizer is a standard HuggingFace fast tokenizer;
        # call it directly (adds <cls>/<eos> via TemplateProcessing automatically).
        encoded = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"].squeeze(0)  # [max_len]

        return {
            "token_ids": token_ids,
            "labels": target_neq,
            "temperature": temp,
        }

class DatasetFactory:
    @staticmethod
    def get_dataset(df, model_name, max_length=1024, masked_value=-100, use_log_scaling=False):
        is_esm3 = "esm3" in model_name.lower()
        
        if is_esm3:
            # CORRECT: Use the official model identifier
            # Loading the model here is fine for metadata, but be careful with VRAM
            tokenizer = EsmSequenceTokenizer()
            dataset = Esm3SequenceDataset(
                df, 
                tokenizer,
                max_length=max_length,
                masked_value=masked_value,
                use_log_scaling=use_log_scaling
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dataset = MdCathSequenceDataset(df, tokenizer, max_length=max_length, masked_value=masked_value, use_log_scaling=use_log_scaling)

        return dataset