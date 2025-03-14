## Description

This repository provides a pipeline for training a **BiLSTM** or **BiLSTM + Self-Attention** model on **ESM** embeddings to classify residues based on Neq values. Training data is from ATLAS library. 

---

## Usage

1. **Prepare the Environment**  
   - All required dependencies all listed in environment.yaml. Create esm_env: `conda env create -f environment.yaml`
   - Activate the environment:  `conda activate esm_env`

2. **Structure**  
   - `scripts/main.py` – Entry point that parses arguments and calls the training loop.  
   - `arguments.py` – Holds the command-line argument definitions.  
   - `train.py` – Implements the training procedure and cross-validation.  
   - `models.py` – Contains model definitions (BiLSTM, attention, FocalLoss, etc.).  
   - `data_utils.py` – Data loading, classification thresholds, custom dataset/collate functions.

3. **Running**  
   From the project’s root (assuming your environment is active):
   ```bash
   python scripts/main.py [options...]

4. **EXample**
  `python main.py   --train_data_file ../../data/train_data.csv   --test_data_file ../../data/test_data.csv  --esm_model esm2_t12_35M_UR50D   --architecture bilstm_attention   --hidden_size 512   --num_layers 3   --dropout 0.3   --loss_function focal   --num_classes 2   --neq_thresholds 1.0   --mixed_precision   --freeze_layers "0-4"   --lr_scheduler reduce_on_plateau   --epochs 20   --patience 5   --batch_size 4 `

## Arguments

- **`--csv_path [str]`** (default: `../neq_training_data.csv`)  
  Path to the CSV with sequences and Neq values.

- **`--esm_model [str]`** (default: `esm2_t12_35M_UR50D`; choices: `esm1_t6_43M_UR50S`, `esm1_t12_85M_UR50S`, `esm1_t34_670M_UR100`, `esm1_t34_670M_UR50D`, `esm1_t34_670M_UR50S`, `esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`, `esm2_t48_15B_UR50D`)  
  Which ESM checkpoint to use.

- **`--batch_size [int]`** (default: `4`)  
  DataLoader batch size.

- **`--epochs [int]`** (default: `20`)  
  Maximum training epochs.

- **`--patience [int]`** (default: `5`)  
  Patience for early stopping.

- **`--lr [float]`** (default: `1e-5`)  
  Learning rate for AdamW.

- **`--weight_decay [float]`** (default: `1e-2`)  
  Weight decay for AdamW.

- **`--num_classes [int]`** (default: `4`)  
  Number of classes for Neq classification.

- **`--neq_thresholds [float list]`** (default: `[1.0, 2.0, 4.0]`)  
  Thresholds for splitting Neq values. Must have `(num_classes - 1)` thresholds.

- **`--architecture [str]`** (default: `bilstm`; choices: `bilstm`, `bilstm_attention`)  
  Model architecture to use.

- **`--hidden_size [int]`** (default: `512`)  
  LSTM hidden size.

- **`--num_layers [int]`** (default: `2`)  
  Number of LSTM layers.

- **`--dropout [float]`** (default: `0.3`)  
  Dropout rate.

- **`--loss_function [str]`** (default: `focal`; choices: `focal`, `crossentropy`)  
  Which loss function to use.

- **`--focal_class_weights`** (flag)  
  Enable class weights (`alpha`) for focal loss if set.

- **`--freeze_layers [str]`** (default: `None`)  
  E.g. `"0-5"` means freeze layers 0..5 in ESM.
  note: all layer norms are kept unfrozen. 

- **`--mixed_precision`** (flag)  
  Enable torch.cuda.amp mixed precision if set.

- **`--lr_scheduler [str]`** (default: `reduce_on_plateau`; choices: `none`, `reduce_on_plateau`)  
  Whether to use a learning-rate scheduler or not.


## Outputs

- **`run_parameters.txt`**  
  Records all the command-line arguments used for the run.

- **`loss_curves_fold_X.png`**  
  Plot of training and validation loss over epochs.

- **`confusion_matrix_fold_X.png`**  
  Confusion matrix for the heldout test set.

- **`classification_report_fold_X.txt`**
  Classification report of the heldout test set.

All these files are placed into a timestamped directory named `run_YYYY-MM-DD_HH-MM-SS`.
