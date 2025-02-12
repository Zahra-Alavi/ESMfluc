"""
Description: This file contains the models for the classifiers with hyperparameter tuning.
Date: 2025-02-07
Author: Ngoc Kim Ngan Tran
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class BaseClassifier:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None
    
    def fit(self):
        """Train the model on the training data."""
        self.model.fit(self.x_train, self.y_train)
    
    def predict(self):
        """Generate predictions on the test data."""
        return self.model.predict(self.x_test)
    
    def evaluate(self, y_pred):
        """Evaluate the model using classification report."""
        return classification_report(self.y_test, y_pred, output_dict=True)

class BaselineClassifier(BaseClassifier):
    def fit(self):
        """No fitting required for baseline."""
        pass  
    
    def predict(self):
        """Predicts the most frequent class in training data."""
        unique, counts = np.unique(self.y_train, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        return np.full_like(self.y_test, fill_value=majority_class)
    
    def evaluate(self, y_pred):
        """Computes simple accuracy for baseline model."""
        print (np.mean(y_pred == self.y_test))

class SklearnClassifier(BaseClassifier):
    def __init__(self, x_train, y_train, x_test, y_test, model):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model = model

class LogisticRegressionClassifier(SklearnClassifier):
    def __init__(self, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if tune_hyperparameters:
            model = self._tune_hyperparameters(x_train, y_train)
        else:
            model = LogisticRegression(solver='liblinear', random_state=42)
        super().__init__(x_train, y_train, x_test, y_test, model)

    def _tune_hyperparameters(self, x_train, y_train):
        """Uses GridSearchCV to find the best hyperparameters for Logistic Regression."""
        param_grid = {
            'max_iter': [100, 200, 300, 400, 500],  # Maximum number of iterations
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l1', 'l2'],  # Regularization type
            'tol': [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criteria
            'class_weight': ['balanced', None],  # Weights associated with classes
            #'n_jobs': [-1],
        }
        grid_search = GridSearchCV(LogisticRegression(solver='liblinear', random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best Logistic Regression Parameters: {grid_search.best_params_}")
        # Saving the best parameters
        if not os.path.exists('../../models'):
            os.makedirs('../../models')
        
        with open('../../models/best_logistic_regression_params.txt', 'w') as f:
            f.write(str(grid_search.best_params_))
        return grid_search.best_estimator_

class RandomForestClassifierModel(SklearnClassifier):
    def __init__(self, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if tune_hyperparameters:
            model = self._tune_hyperparameters(x_train, y_train)
        else:
            model = RandomForestClassifier(random_state=42)
        super().__init__(x_train, y_train, x_test, y_test, model)

    def _tune_hyperparameters(self, x_train, y_train):
        """Uses GridSearchCV to find the best hyperparameters for Random Forest."""
        param_grid = {
            'n_estimators': [10, 50, 100, 200],  # Number of trees
            'max_depth': [None, 10, 20, 30],  # Tree depth
            'min_samples_split': [2, 5, 10],  # Min samples to split a node
            'min_samples_leaf': [1, 2, 4],  # Min samples at a leaf node
            'bootstrap': [True, False],  # Whether to use bootstrap sampling
            'n_jobs': [-1],
            'class_weight': ['balanced', None, 'balanced_subsample'],  # Weights associated with classes
            'criterion': ['gini', 'entropy'],  # Splitting criteria
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best Random Forest Parameters: {grid_search.best_params_}")
        
        # Saving the best parameters
        if not os.path.exists('../../models'):
            os.makedirs('../../models')
        
        with open('../../models/best_random_forest_params.txt', 'w') as f:
            f.write(str(grid_search.best_params_))
        return grid_search.best_estimator_
