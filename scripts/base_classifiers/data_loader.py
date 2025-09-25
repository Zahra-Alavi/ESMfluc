"""
Description: This file contains the DataLoader class which is used to load the data from the csv file and split it into training, validation and test sets.
Date: 2025-02-07

"""

import pandas as pd
from decimal import Decimal
from sklearn.model_selection import train_test_split
from features import *

class DataLoader:
    def __init__(self, file_path, feature_engineering_version, esm_model_name, binary_classification=False):
        print("Loading data...")
        self.data = pd.read_csv(file_path)
        self.sequences = self.data['sequence']
        self.neq_values = self.data['neq'].apply(lambda x: x.replace('[', '').replace(']', '').split(','))
        self.neq_values = self.neq_values.apply(lambda x: [Decimal(i) for i in x])
        self.binary_classification = binary_classification
        
        # Feature extraction
        feature_extractor_class = globals().get(f"FeatureExtraction{feature_engineering_version.replace('.', '_')}", None)
        if feature_extractor_class is None:
            raise ValueError(f"Invalid feature engineering version: {feature_engineering_version}")

        if feature_engineering_version == "1.3":
            self.feature_extractor = feature_extractor_class(esm_model_name)
        else:
            self.feature_extractor = feature_extractor_class()
        self.features, self.targets = self.feature_extractor.extract_features(self.sequences, self.neq_values)

    def get_data(self):
        if self.binary_classification:
            self.targets = self.classify_neq(self.targets)
        return self.features, self.targets
    
    def classify_neq(self, neq_values):
        return [0 if neq == Decimal("1.0") else 1 for neq in neq_values]