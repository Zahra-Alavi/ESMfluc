# BASE CLASSIFIERS

## Description

This script is used to run base classifiers (Logistic Regression, Random Forest Classifiers, Conditional Random Field) on a dataset for predicting Neq values of amino acids. It includes functionalities for data learning, feature engineering, regression, and classification tasks.

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

### Logistic Regression and Random Forest Classifiers

Use `classifiers_runner.py` to run the script with various options to perform specific tasks for **Logistic Regression and Random Forest Classifiers only**.

#### General Syntax

```
python classfiers_runner.py [OPTIONS]
```

#### Available Arguments

-   `--data_learning`: Perform data learning, including histograms and amino acid analysis
-   `--model {LogisticRegressionClassifier, RandomForestClassifier}`: Choose the classification model to run
-   `--all`: Run all tasks including data learning, Logistic Regression Classifier model, and Random Forest Classifier
-   `--feature_enineering_version <version>`: Specify feature engineering version (default: 1.1). Currently, we have 1.1, 1.2, and 1.3.
-   `--hyperparameter_tuning <True/False>`: Enable hyperparameter tuning for the classifiers (default: False)
-   `--esm_model_learning <True/False>`: Enable ESM model learning to find the best ESM model (default: False)
-   `--esm_model <model_name>`: For feature engineering version 1.3, we use

#### Example Commands

1. Run all tasks:

    `python classifier_runner.py --all`

2. Perform data learning:

    `python classifier_runner.py --data_learning`

3. Run a specific classifier (Random Forest):

    `python classifier_runner.py --model RandomForestClassifier`

4. Run a classifier with ESM embedding:
   `python classifier_runner.py --model RandomForestClassifier --feature_engineering_version 1.3`

5. Run a classifier with hyperparameter tuning:

    `python classifier_runner.py --model LogisticRegressionClassifier --hyperameter_tuning True`

6. Run ESM model learning:

    `python classifier_runner.py --esm_model_learning True`

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
