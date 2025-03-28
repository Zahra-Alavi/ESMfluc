# Overview

The pipeline has been modified to:

- Save attention weights from the self-attention layer of the BiLSTM model.

- Save the best model for inference.

- Support DataParallel wrapping, enabling the use of a heavier embedding model with a larger batch size.

- Multihead Attention for BiLSTM Model

# Folder Structure

- `main.py`: The main entry point for training and evaluation.

- `arguments.py`: Defines all available command-line arguments.

- `train2.py`: Handles training and evaluation logic.

- `models.py`: Contains model definitions.

- `data_utils.py`: Preprocessing and data handling utilities.

- `analysis.py`: Scripts for performing histogram analysis from the input results.

- `attention_visualizer.py`: Plots attention heatmaps for each sequence.

# Training Instructions

Example:

```bash
python main.py \
  --train_data_file ../../data/train_data.csv \
  --test_data_file ../../data/test_data.csv \
  --esm_model esm2_t33_650M_UR50D \
  --architecture bilstm_attention \
  --dropout 0.3 \
  --loss_function focal \
  --mixed_precision \
  --lr_scheduler reduce_on_plateau \
  --epochs 80 \
  --patience 3 \
  --batch_size 2 \
  --num_classes 2 \
  --neq_thresholds 1.0 \
  --freeze_layers '0-4' \
  --data_parallel \
  --num_layers 3
```

# Visualizing Attention Heatmaps

Once the model is trained and saved, you can apply it to sequences from a FASTA file and generate attention heatmaps using:

```bash python attention_visualizer_interactive.py --checkpoint best_model_shuffled.pth --fasta_file examples.fasta ```

# Generating Histogram Analysis

1. Generate `results.csv` file

After training the model, you can generate the `results.csv` file using a similar command. Since the best model is already saved during training, you only need to specify the folder containing the saved model. The results will automatically be saved in the `results/` folder.

**Argument**: --result_foldername (specifies the folder name where results are stored)

Ex: 

```bash
python main.py \
  --train_data_file ../../data/train_data.csv \
  --test_data_file ../../data/test_data.csv \
  --esm_model esm2_t33_650M_UR50D \
  --architecture bilstm_attention \
  --dropout 0.3 \
  --loss_function focal \
  --mixed_precision \
  --lr_scheduler reduce_on_plateau \
  --epochs 80 \
  --patience 3 \
  --batch_size 2 \
  --num_classes 2 \
  --neq_thresholds 1.0 \
  --freeze_layers '0-4' \
  --data_parallel \
  --num_layers 3 \
  --result_foldername fold_1
```

2. Generate Histograms from `results.csv`

Once the results are generated, you can visualize them by running the following command:

```
python analysis.py --folder ../../results/[folder_name]
```
Replace [`folder_name`] with the actual folder where your results are stored.