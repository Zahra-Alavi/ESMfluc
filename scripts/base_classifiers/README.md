# BASE CLASSIFIERS AND REGRESSORS

## Description

This module provides implementations for both classification and regression tasks for predicting Neq values of amino acids. It includes:

**Classification Models:**
- Logistic Regression Classifier (for binary classification: flexible vs. non-flexible)
- Random Forest Classifier
- Conditional Random Field (CRF)

**Regression Models:**
- Linear Regression (for continuous Neq value prediction)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Random Forest Regressor

All models support:
- Multiple feature engineering versions (1.1, 1.2, 1.3)
- Hyperparameter tuning with GridSearchCV
- ESM embeddings for feature extraction

## Classification vs. Regression

**Use Classification models when:**
- You want to predict discrete categories (e.g., flexible vs. non-flexible amino acids)
- Binary classification: Neq == 1.0 (non-flexible) vs. Neq != 1.0 (flexible)

**Use Regression models when:**
- You want to predict continuous Neq values (e.g., 1.0, 2.86, 3.45, etc.)
- You need precise flexibility predictions rather than binary categories
- You want to capture the full range of flexibility values

## Prerequisites

Before running the script, ensure you have Python installed and set up a virtual environment with the required dependencies.

1. Create and activate a virutal environment

```
python -m venv <folder-name>
source venv/bin/activate # On macOS/Linux
venv\Scripts\activate    # On Windows
```

2. Install dependencies

```
pip install -r requirements.txt
```

## Usage

### Classifiers and Regressors

Use `classifiers_runner.py` to run the script with various options to perform specific tasks for **Classification or Regression models**.

#### General Syntax

```
python classifiers_runner.py [OPTIONS]
```

#### Available Arguments

-   `--data_learning`: Perform data learning, including histograms and amino acid analysis
-   `--model {LogisticRegressionClassifier, RandomForestClassifier, LinearRegression, RidgeRegression, LassoRegression, RandomForestRegressor}`: Choose the classification or regression model to run
-   `--all`: Run all tasks including data learning, Logistic Regression Classifier model, and Random Forest Classifier
-   `--feature_engineering_version <version>`: Specify feature engineering version (default: 1.1). Currently, we have 1.1, 1.2, and 1.3.
-   `--hyperameter_tuning`: Enable hyperparameter tuning for the classifiers/regressors (default: False)
-   `--esm_model_learning`: Enable ESM model learning to find the best ESM model (default: False)
-   `--esm_model <model_name>`: For feature engineering version 1.3, specify the ESM model to use

#### Example Commands

##### Classification Examples

1. Run all tasks:

    `python classifiers_runner.py --all`

2. Perform data learning:

    `python classifiers_runner.py --data_learning`

3. Run a specific classifier (Random Forest):

    `python classifiers_runner.py --model RandomForestClassifier`

4. Run a classifier with ESM embedding:
   `python classifiers_runner.py --model RandomForestClassifier --feature_engineering_version 1.3`

5. Run a classifier with hyperparameter tuning:

    `python classifiers_runner.py --model LogisticRegressionClassifier --hyperameter_tuning`

6. Run ESM model learning:

    `python classifiers_runner.py --esm_model_learning`

##### Regression Examples

1. Run Linear Regression for continuous Neq prediction:

    `python classifiers_runner.py --model LinearRegression --feature_engineering_version 1.1`

2. Run Ridge Regression with hyperparameter tuning:

    `python classifiers_runner.py --model RidgeRegression --hyperameter_tuning`

3. Run Lasso Regression:

    `python classifiers_runner.py --model LassoRegression`

4. Run Random Forest Regressor with ESM embeddings:

    `python classifiers_runner.py --model RandomForestRegressor --feature_engineering_version 1.3`

## Model Performance

### Regression Metrics

Regression models are evaluated using:
- **MSE** (Mean Squared Error): Average squared difference between predicted and actual values
- **RMSE** (Root Mean Squared Error): Square root of MSE, in the same units as Neq
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual values
- **R²** (R-squared): Proportion of variance explained by the model (0-1, higher is better)

### Hyperparameter Tuning

Hyperparameter tuning uses GridSearchCV with 5-fold cross-validation for classifiers and 3-5 fold for regressors. The optimized parameter grids focus on the most impactful hyperparameters to balance performance and computational efficiency:

- **Ridge/Lasso**: Alpha values, solver selection, max iterations
- **Random Forest**: Number of estimators, max depth, min samples split/leaf, max features
- **Logistic Regression**: Regularization strength (C), penalty type, max iterations

### Conditional Random Field

Use `conditional_random_field.py` to run the script with various options to perform specific tasks for **Conditional Random Field only**.

#### General Syntax

```
python conditional_random_field.py [OPTIONS]
```

#### Available Arguments

-   `--train_data_file`: The path to the train data CSV file, default to ../../data/train_data.csv
-   `--test_data_file`: The path to the test data CSV file, default to ../../data/train_data.csv
-   `--amino_acids_file`: Path to the amino acids characteristics CSV file, default to ../../data/train_data.csv
-   `--window_size`: Window size for feature extraction (default is 2)
-   `--features`: Comma-separated list of features to use (choices: charges, polar, hydrophobic, molecular_weight, pKa, pKb, pKx, pI).
-   `--hyperparameter_tuning`: Include this argument if wants to perform hyperparameter tuning

### Notes

-   Ensure that the dataset file `../../data/train_data.csv` and `../../data/test_data.csv` is available in the expected location.
