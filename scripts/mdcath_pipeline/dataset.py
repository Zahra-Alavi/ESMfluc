import torch
from torch.utils.data import Dataset
import ast

class MdCathSequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024, masked_value = -100):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.masked_value = masked_value
        # For mdcath, we have different temperatures, but for atlas, we don't, so we have to handle it dynamically.
        self.temp_cols = [c for c in dataframe.columns if c.startswith('neq')]
        if self.temp_cols:
            if self.temp_cols[0].startswith('neq_'):
                self.temperatures = [float(col.split('_')[1]) for col in self.temp_cols]
            else:
                self.temperatures = [300.0] #atlas collects neq at 300K
        self.max_temp = max(self.temperatures)
        self.min_temp = min(self.temperatures)
        
    def __len__(self):
        # Each domain has multiple temperature, so the total number of samples is number of domains * number of temperatures
        return len(self.df) * len(self.temp_cols)
    
    def __getitem__(self, idx):
        # Because the length is number of domains * number of temperatures, we need to figure out which domain and which temperature this index corresponds to
        domain_idx = idx // len(self.temp_cols)
        temp_idx = idx % len(self.temp_cols)
        
        row = self.df.iloc[domain_idx]
        temp = self.temperatures[temp_idx]
        if len(self.temperatures) > 1:
            temp = (float(temp) - self.min_temp) / (self.max_temp - self.min_temp)
        
        sequence = row['sequence']
        labels = ast.literal_eval(row[self.temp_cols[temp_idx]])  # Convert string representation of list to actual list
        
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Padding labels
        target_neq = torch.full((self.max_len,), self.masked_value)
        actual_len = min(len(labels), self.max_len - 2)  # Account for [CLS] and [SEP]
        target_neq[1:actual_len+1] = torch.tensor(labels[:actual_len])
        
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_neq,
            'temperature': torch.tensor(temp, dtype=torch.float)
        }