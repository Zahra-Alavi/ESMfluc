# Overview

This folder provides comprehensive pipeline for predicting the flexibility of amino acids in a protein sequence. The core of the system leverages state-of-the-art protein language models (PLMs) form the ESM family to generate rich, contextual embeddings, which are then used to train various downstream classification architectures.

- **Modular Architecture**: Support for different model architectures (bilstm, bilstm_attention, transformer, esm_linear)
- **Advanced Loss Functions**: Implementation of focal and crossentropy loss, including support for balanced class weights. The NC (Normalized Centroid) loss is also included for enhanced training.
- **Handling Data Imbalance**: Built-in support for oversampling and undersampling to manage imbalanced datasets.
- **Scalability**: Support for multi-GPU training via nn.DataParallel and mixed-precision training for faster, more efficient model training.
- **Inference and Analysis**: The pipeline allows for saving the best-performing model for inference and includes a dedicated script for extracting attention weights and visualizing them.

# Folder Structure

Training:

- `main.py`: The main entry point for training and evaluation.

- `arguments.py`: Defines all available command-line arguments.

- `train.py`: Handles training and evaluation logic.

- `models.py`: Contains model definitions.

- `data_utils.py`: Preprocessing and data handling utilities.

- `nc_losses.py`: Implementing NC Loss

Inference and Analysis:

- `get_attn.py`: Main inference scripts, takes a fasta file and the best model check point, returns a JSON file with sequences, their attention weights and predicted neq class.
- `pheatmap_functions.py`: Contains functions to plot attention heat maps and perform PCA. It can also analyze variant effect for an input file including a WT and mutants.  

# Training Instructions
To train a model, use the `main.py` script and pass the desired arguments.

**Example Command:**

The following command demonstrates how to train an `esm2_t33_650M_UR50D` model with a BiLSTM, using focal loss and mixed precision.

```bash
python main.py \
    --esm_model esm2_t33_650M_UR50D \
    --hidden_size 512 \
    --num_layers 3 \
    --dropout 0.3 \
    --lr_scheduler reduce_on_plateau \
    --epochs 80 \
    --patience 3 \
    --batch_size 2 \
    --freeze_layers 0-4 \
    --loss_mode supervised \
    --loss_function focal \
    --head softmax \
    --num_classes 2 \
    --neq_thresholds 1.0 \
    --train_data_file ./train_data.csv \
    --test_data_file ./test_data.csv \
    --device cuda
```
# Inference Instructions

Once the model is trained and saved, you can apply it to sequences from a FASTA file to get predicted neq class and attention weights. The results will be saved in your desired path as a JSON file.

```bash 
python get_attn.py --checkpoint best_model.pth --fasta_file your_sequences.fasta your_ss.csv --ouput path_to_the_final_output
```

**Note**: You can optionally provide a secondary structure prediction CSV file from **NetSurfP** using the `--ss_csv` argument to include secondary structure annotations in your output.

If a CSV file for secondary structure prediction is given, the final JSON will also include ss_pred. Such CSV file can be obtained from: https://services.healthtech.dtu.dk/services/NetSurfP-3.0/. This is only useful if you later want to plot attention heatmaps with ss annotations.

Once you run inference, you can visualize the attentin mechanism using `pheatmap_functions.py`

# Command Line Arguments

Here is the full list of available command-line arguments:

| Argument                           | Type        | Default                     | Description                                                   |
|------------------------------------|-------------|-----------------------------|---------------------------------------------------------------|
| `--train_data_file`                | str         | `../../data/train_data.csv` | Path to the training data.                                    |
| `--test_data_file`                 | str         | `../data/test_data.csv`     | Path to the test data.                                        |
| `--esm_model`                      | str         | `esm2_t12_35M_UR50D`        | The ESM checkpoint to use.                                    |
| `--batch_size`                     | int         | `4`                         | Batch size for training.                                      |
| `--epochs`                         | int         | `20`                        | Number of training epochs.                                    |
| `--patience`                       | int         | `5`                         | Early stopping patience.                                      |
| `--lr`                             | float       | `1e-5`                      | Learning rate for the AdamW optimizer.                        |
| `--weight_decay`                   | float       | `1e-2`                      | Weight decay for the AdamW optimizer.                         |
| `--num_classes`                    | int         | `4`                         | Number of target classes.                                     |
| `--neq_thresholds`                  | list[float] | `[1.0, 2.0, 4.0]`           | Thresholds for Neq classification.                            |
| `--architecture`                   | str         | `bilstm`                    | Model architecture choice.                                    |
| `--hidden_size`                    | int         | `512`                       | Hidden size of the LSTM layers.                               |
| `--num_layers`                     | int         | `2`                         | Number of LSTM layers.                                        |
| `--dropout`                        | float       | `0.3`                       | Dropout rate.                                                 |
| `--transformer_nhead`              | int         | `8`                         | Number of attention heads.                                    |
| `--transformer_num_encoder_layers` | int         | `6`                         | Number of Transformer encoder layers.                         |
| `--transformer_dim_feedforward`    | int         | `1024`                      | Feedforward dimension in Transformer.                         |
| `--bidirectional`                  | int         | `1`                         | Use a bidirectional LSTM (`1`) or not (`0`).                  |
| `--loss_function`                  | str         | `focal`                     | The loss function to use (`focal` or `crossentropy`).          |
| `--focal_class_weights`            | bool        | `False`                     | Whether to use class weights with focal loss.                 |
| `--head`                           | str         | `softmax`                   | The classification head architecture.                         |
| `--lambda_nc1`                     | float       | `1.0`                       | NC loss parameter.                                            |
| `--beta_nc2`                       | float       | `0.5`                       | NC loss parameter.                                            |
| `--lambda_ce`                      | float       | `1.0`                       | Cross-entropy loss parameter.                                 |
| `--loss_mode`                      | str         | `both`                      | Loss composition (`supervised`, `nc`, or `both`).              |
| `--oversampling`                   | bool        | `False`                     | Enable sequence-level oversampling.                           |
| `--oversampling_threshold`         | float       | `0.1`                       | Fraction of minority residues required for oversampling.       |
| `--undersampling_threshold`        | float       | `0.01`                      | Undersampling threshold for down-weighting.                   |
| `--undersampling_intensity`        | float       | `0.1`                       | Scaling factor for undersampling.                             |
| `--oversampling_intensity`         | float       | `5.0`                       | Scaling factor for oversampling.                              |
| `--data_parallel`                  | bool        | `False`                     | Use `nn.DataParallel` for multi-GPU training.                 |
| `--warmup_epochs`                  | int         | `0`                         | Epochs to keep `lambda_ce > 0`.                               |
| `--freeze_all_backbone`            | bool        | `False`                     | Freeze all ESM backbone parameters.                           |
| `--freeze_layers`                  | str         | `None`                      | Specify ESM layers to freeze (e.g., `0-5`).                   |
| `--mixed_precision`                | bool        | `False`                     | Enable mixed precision training.                              |
| `--lr_scheduler`                   | str         | `reduce_on_plateau`         | Learning rate scheduler choice.                               |
| `--device`                         | str         | `cuda`                      | PyTorch device to use.                                        |
| `--amp_dtype`                      | str         | `fp16`                      | Autocast dtype for mixed precision.                           |
| `--seed`                           | int         | `42`                        | Global random seed for reproducibility.                       |
| `--result_foldername`              | str         | `timestamp`                 | Name for the result folder.                                   |
