"""
Description: This file contains the feature extraction functions for the project.
Date: 2025-02-07
Author: Ngoc Kim Ngan Tran
"""

import torch
import esm
import pandas as pd
from decimal import Decimal
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

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
            row['Amino Acids']: [row['Charges'], row['Polar'], row['Hydrophobic']] 
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
        self.model_name = model_name
        self._load_esm_model(model_name)

    def _load_esm_model(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.repr_layers = int(model_name.split("_")[1].replace("t", ""))
        if model_name == "esm2_t48_15B_UR50D":
            # offload the model to CPU to reduce GPU memory usage. Ex: https://github.com/facebookresearch/esm/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py

            # init the distributed world with world_size 1
            print("Offload the model to CPU to reduce GPU memory usage")
            url = "tcp://localhost:23456"
            torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)
            
            model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)
            
            # initialize the model with FSDP wrapper
            fsdp_params = dict(
                mixed_precision=True,
                flatten_parameters=True,
                state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
                cpu_offload=True,  # enable cpu offloading
            )
            
            with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
                model, vocab = esm.pretrained.load_model_and_alphabet_core(
                    model_name, model_data, regression_data
                )
                self.batch_converter = vocab.get_batch_converter()
                model.eval()

                # Wrap each layer in FSDP separately
                for name, child in model.named_children():
                    if name == "layers":
                        for layer_name, layer in child.named_children():
                            wrapped_layer = wrap(layer)
                            setattr(child, layer_name, wrapped_layer)
                self.model = wrap(model)
                print("Model wrapped with FSDP")
        else:
            self.model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval()
            self.model.to(self.device)
    
    def extract_features(self, sequences, neq_values):
        print("Feature extraction version 1.3")
        
        seq_embedding, targets = [], []
        for i, seq in enumerate(sequences):
            data = [(f"protein_{i}", seq)]
            labels, strs, tokens = self.batch_converter(data)
            print("Self.device:", self.device)
            with torch.no_grad():
                results = self.model(tokens.to(self.device), repr_layers=[self.repr_layers])
                token_embedding = results["representations"][self.repr_layers]
                # If model is ESM 2 then token embedding is 1:-1 for second dimension, else 1: for esm 1
                if self.model_name.startswith("esm1"):
                    residue_embeddings = token_embedding[0, 1:]
                else:
                    residue_embeddings = token_embedding[0, 1:-1]
                seq_embedding.extend(residue_embeddings.cpu().numpy())
                targets.extend(neq_values[i])
        return seq_embedding, targets