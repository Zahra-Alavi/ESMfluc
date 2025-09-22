"""
Description: This file contains the feature extraction functions for the project.
Date: 2025-02-07
Author: Ngoc Kim Ngan Tran
"""

import torch
import esm
import pandas as pd
from transformers import AutoTokenizer, AutoModel

class BaseFeatureExtraction:
    def __init__(self):
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Standard amino acids

    def _one_hot_encode(self, aa):
        aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        aa_encoded = [0] * len(self.amino_acids)
        if aa in aa_to_idx:
            aa_encoded[aa_to_idx[aa]] = 1
        return aa_encoded

    def extract_features(self, sequences, neq_values):
        raise NotImplementedError("Subclasses should implement this method")

class FeatureExtraction1_0(BaseFeatureExtraction):
    def extract_features(self, sequences, neq_values):
        print("Feature extraction version 1.0")
        features, targets = [], []
        for seq, neq_seq in zip(sequences, neq_values):
            for i, aa in enumerate(seq):
                aa_encoded = self._one_hot_encode(aa)
                position = i / len(seq)
                prev_neq = float(neq_seq[i - 1]) if i > 0 else 0.0
                next_neq = float(neq_seq[i + 1]) if i < len(seq) - 1 else 0.0
                features.append(aa_encoded + [prev_neq, next_neq, position])
                targets.append(float(neq_seq[i]))
        return features, targets

class FeatureExtraction1_1(BaseFeatureExtraction):
    def extract_features(self, sequences, neq_values):
        print("Feature extraction version 1.1")
        features, targets = [], []
        for seq, neq_seq in zip(sequences, neq_values):
            for i, aa in enumerate(seq):
                aa_encoded = self._one_hot_encode(aa)
                position = i / len(seq)
                prev_aa = seq[i - 1] if i > 0 else None
                next_aa = seq[i + 1] if i < len(seq) - 1 else None
                prev_aa_encoded = self._one_hot_encode(prev_aa) if prev_aa else [0] * len(self.amino_acids)
                next_aa_encoded = self._one_hot_encode(next_aa) if next_aa else [0] * len(self.amino_acids)
                features.append(aa_encoded + prev_aa_encoded + next_aa_encoded + [position])
                targets.append(float(neq_seq[i]))
        return features, targets

class FeatureExtraction1_2(FeatureExtraction1_1):
    def __init__(self):
        super().__init__()
        self.amino_acids_characteristics = pd.read_csv('../../data/amino_acids_characteristics.csv')
        self.amino_acids_characteristics.columns = self.amino_acids_characteristics.columns.str.strip()
        self.characteristics_dict = {
            row['amino_acids']: [row['charges'], row['polar'], row['hydrophobic']] 
            for _, row in self.amino_acids_characteristics.iterrows()
        }
    
    def extract_features(self, sequences, neq_values):
        print("Feature extraction version 1.2")
        features, targets = super().extract_features(sequences, neq_values)
        features_index = 0
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in self.characteristics_dict:
                    features[features_index] += self.characteristics_dict[aa]
                features_index += 1
        return features, targets

class FeatureExtraction1_3(BaseFeatureExtraction):
    def __init__(self, model_name):
        super().__init__()
        print("ESM embedding with ESM model:", model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}", use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
    
    def extract_features(self, sequences, neq_values):
        print("Feature extraction version 1.3")
        
        seq_embedding, targets = [], []
        for i, seq in enumerate(sequences):
            inputs = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # use last hidden state as embeddings
                token_embeddings = outputs.last_hidden_state.squeeze(0)

                # skip special tokens ([CLS], [SEP]) if present
                if self.tokenizer.cls_token_id is not None and self.tokenizer.sep_token_id is not None:
                    residue_embeddings = token_embeddings[1:-1]
                else:
                    residue_embeddings = token_embeddings

                seq_embedding.extend(residue_embeddings.cpu().numpy())
                targets.extend(neq_values[i])
        
        return seq_embedding, targets