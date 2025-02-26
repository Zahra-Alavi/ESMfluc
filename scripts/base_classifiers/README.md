# BASE CLASSIFIERS

## Description
This script is used to run base classifiers (Logistic Regression, Random Forest Classifiers) on a dataset for predicting Neq values of amino acids. It includes functionalities for data learning, feature engineering, regression, and classification tasks.

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

Use `classifiers_runner.py` to run the script with various options to perform specific tasks. 

### General Syntax
```
python classfiers_runner.py [OPTIONS]
```

### Available Arguments

- `--data_learning`: Perform data learning, including histograms and amino acid analysis
- `--model {LogisticRegressionClassifier, RandomForestClassifier}`: Choose the classification model to run
- `--all`: Run all tasks including data learning, Logistic Regression Classifier model, and Random Forest Classifier
- `--feature_enineering_version <version>`: Specify feature engineering version (default: 1.1). Currently, we have 1.0, 1.1, 1.2, and 1.3.
    - 1.0: cheated feature engineering (wouldn't recommended to use this) 
- `--hyperparameter_tuning <True/False>`: Enable hyperparameter tuning for the classifiers (default: False)
- `--esm_model_learning <True/False>`: Enable ESM model learning to find the best ESM model (default: False)
- `--esm_model <model_name>`: For feature engineering version 1.3, we use hyper 

### Example Commands

1. Run all tasks:

    ```python script.py --all```

2. Perform data learning:

    ```python script.py --data_learning```

3. Run a specific classifier (Random Forest):

    ```python script.py --model RandomForestClassifier```

4. Run a classifier with hyperparameter tuning:

    ```python script.py --model LogisticRegressionClassifier --hyperameter_tuning True```

5. Run ESM model learning:

    ```python script.py --esm_model_learning True```

### Notes

- Ensure that the dataset file `../../data/neq_training_data.csv` is available in the expected location.