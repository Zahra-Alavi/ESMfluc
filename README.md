# The Effects of Embedding Models ESM on Predicting the Flexibility of Amino Acids

This project investigates the effects of ESM (Evolutionary Scale Modeling) embeddings on predicting the flexibility of amino acids. The pipeline processes amino acid sequence data, applies feature engineering (e.g., one-hot encoding or embeddings), and trains models to predict flexibility based on Neq values. The main entry point for running the project is `scripts/main.py`.

## Preprequisties

To run this project, ensure you have the folowing installed:
- Python 3.8 or higher
- Virtualenv (optional but recommended)

## Setup Instructions 

### 1. Clone the Repository

Clone this repositiory to your local machine:

```
git clone https://github.com/Zahra-Alavi/ESMfluc.git
cd ESMfluc
```

### 2. Set Up a Python Environment

Create and activate a virtual environment to manage dependencies:

#### Create the Environment
```
python -m venv venv
```
*Note: If you choose a different folder name besides `venv`, remember to add it to your `.gitignore` file to avoid committing it to the repository.*

#### Activate the Environment

```
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Required Libraries

Install the required Python packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```

### 3. Prepare Data

Ensure your dataset (e.g., `neq_original_data.csv`) is placed in the `data/` directory. The default path in the code is `../data/neq_original_data.csv`. If your data is located elsewhere, specify the path using the `--data` argument when running the script.

## Usage

To run the project, please navigate through the different folders inside the `scripts/` directory:

  - The `base_classifiers` folder contains code for the Logistic Regression and Random Forest Classification models.

  - The `conditional_random_fields` folder contains code for the CRF model.

  - The `attention_visualizing` folder contains the latest code for the BiLSTM model.

Please refer to the instructions provided within each folder for details on how to use the code.
