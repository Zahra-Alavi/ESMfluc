"""
Description: This file contains the models for the classifiers with hyperparameter tuning.
Date: 2025-02-07
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score

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
        return (np.mean(y_pred == self.y_test))

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
            'max_iter': [200, 500],
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None],
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
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None],
            'criterion': ['gini', 'entropy'],
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

class BaseRegressor:
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
        """Evaluate the model using regression metrics."""
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }

class SklearnRegressor(BaseRegressor):
    def __init__(self, x_train, y_train, x_test, y_test, model):
        super().__init__(x_train, y_train, x_test, y_test)
        self.model = model

class LinearRegressionModel(SklearnRegressor):
    def __init__(self, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if tune_hyperparameters:
            model = self._tune_hyperparameters(x_train, y_train)
        else:
            model = LinearRegression()
        super().__init__(x_train, y_train, x_test, y_test, model)
    
    def _tune_hyperparameters(self, x_train, y_train):
        """Uses GridSearchCV to find the best hyperparameters for Linear Regression."""
        param_grid = {
            'fit_intercept': [True, False],
            'positive': [True, False],
        }
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best Linear Regression Parameters: {grid_search.best_params_}")
        
        # Saving the best parameters
        if not os.path.exists('../../models'):
            os.makedirs('../../models')
        
        with open('../../models/best_linear_regression_params.txt', 'w') as f:
            f.write(str(grid_search.best_params_))
        return grid_search.best_estimator_

class RidgeRegressionModel(SklearnRegressor):
    def __init__(self, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if tune_hyperparameters:
            model = self._tune_hyperparameters(x_train, y_train)
        else:
            model = Ridge(random_state=42)
        super().__init__(x_train, y_train, x_test, y_test, model)
    
    def _tune_hyperparameters(self, x_train, y_train):
        """Uses GridSearchCV to find the best hyperparameters for Ridge Regression."""
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'solver': ['auto', 'lsqr', 'sag'],
            'max_iter': [2000],
        }
        grid_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best Ridge Regression Parameters: {grid_search.best_params_}")
        
        # Saving the best parameters
        if not os.path.exists('../../models'):
            os.makedirs('../../models')
        
        with open('../../models/best_ridge_regression_params.txt', 'w') as f:
            f.write(str(grid_search.best_params_))
        return grid_search.best_estimator_

class LassoRegressionModel(SklearnRegressor):
    def __init__(self, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if tune_hyperparameters:
            model = self._tune_hyperparameters(x_train, y_train)
        else:
            model = Lasso(random_state=42, max_iter=10000)
        super().__init__(x_train, y_train, x_test, y_test, model)
    
    def _tune_hyperparameters(self, x_train, y_train):
        """Uses GridSearchCV to find the best hyperparameters for Lasso Regression."""
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'max_iter': [10000],
            'selection': ['cyclic', 'random'],
        }
        grid_search = GridSearchCV(Lasso(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best Lasso Regression Parameters: {grid_search.best_params_}")
        
        # Saving the best parameters
        if not os.path.exists('../../models'):
            os.makedirs('../../models')
        
        with open('../../models/best_lasso_regression_params.txt', 'w') as f:
            f.write(str(grid_search.best_params_))
        return grid_search.best_estimator_

class RandomForestRegressorModel(SklearnRegressor):
    def __init__(self, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if tune_hyperparameters:
            model = self._tune_hyperparameters(x_train, y_train)
        else:
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        super().__init__(x_train, y_train, x_test, y_test, model)
    
    def _tune_hyperparameters(self, x_train, y_train):
        """Uses GridSearchCV to find the best hyperparameters for Random Forest Regressor."""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best Random Forest Regressor Parameters: {grid_search.best_params_}")
        
        # Saving the best parameters
        if not os.path.exists('../../models'):
            os.makedirs('../../models')
        
        with open('../../models/best_random_forest_regressor_params.txt', 'w') as f:
            f.write(str(grid_search.best_params_))
        return grid_search.best_estimator_

class ClassifierFactory:
    @staticmethod
    def get_classifier(model_name, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if model_name == "RandomForestClassifier":
            return RandomForestClassifierModel(x_train, y_train, x_test, y_test, tune_hyperparameters)
        elif model_name == "LogisticRegressionClassifier":
            return LogisticRegressionClassifier(x_train, y_train, x_test, y_test, tune_hyperparameters)
        else:
            raise ValueError(f"Unknown model: {model_name}")

class RegressorFactory:
    @staticmethod
    def get_regressor(model_name, x_train, y_train, x_test, y_test, tune_hyperparameters=False):
        if model_name == "LinearRegression":
            return LinearRegressionModel(x_train, y_train, x_test, y_test, tune_hyperparameters)
        elif model_name == "RidgeRegression":
            return RidgeRegressionModel(x_train, y_train, x_test, y_test, tune_hyperparameters)
        elif model_name == "LassoRegression":
            return LassoRegressionModel(x_train, y_train, x_test, y_test, tune_hyperparameters)
        elif model_name == "RandomForestRegressor":
            return RandomForestRegressorModel(x_train, y_train, x_test, y_test, tune_hyperparameters)
        else:
            raise ValueError(f"Unknown model: {model_name}")
