"""
Description: This file contains the models for the classifiers.
Date: 2025-02-07
Author: Ngoc Kim Ngan Tran
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class BaseClassifier:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None
    
    def fit(self):
        self.model.fit(self.x_train, self.y_train)
    
    def predict(self):
        return self.model.predict(self.x_test)
    
    def evaluate(self, y_pred):
        return classification_report(self.y_test, y_pred, output_dict=True)

class BaselineClassifier(BaseClassifier):
    def fit(self):
        pass  # No fitting required for baseline
    
    def predict(self):
        majority_class = np.argmax(np.bincount(self.y_train))
        return np.full_like(self.y_test, fill_value=majority_class)
    
    def evaluate(self, y_pred):
        return np.mean(y_pred == self.y_test)

class SklearnClassifier(BaseClassifier):
    def __init__(self, x_train, y_train, x_test, y_test, model):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model = model

class LogisticRegressionClassifier(SklearnClassifier):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test, LogisticRegression(solver='liblinear', random_state=42))

class RandomForestClassifierModel(SklearnClassifier):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test, RandomForestClassifier(random_state=42))
        