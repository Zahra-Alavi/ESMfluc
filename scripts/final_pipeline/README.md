# Overview

The pipeline has been modified to:

- Save attention weights from the self-attention layer of the BiLSTM model.

- Save the best model for inference.

- Support DataParallel wrapping, enabling the use of a heavier embedding model with a larger batch size.

- Multihead Attention for BiLSTM Model

# Folder Structure

Training:

- `main.py`: The main entry point for training and evaluation.

- `arguments.py`: Defines all available command-line arguments.

- `train2.py`: Handles training and evaluation logic.

- `models.py`: Contains model definitions.

- `data_utils.py`: Preprocessing and data handling utilities.

Inference and Analysis:

- `get_attn.py`: Main inference scripts, takes a fasta file and the best model check point, returns a JSON file with sequences, their attention weights and predicted neq class.
- `pheatmap_functions.py`: Contains functions to plot attention heat maps and perform PCA. It can also analyze variant effect for an input file including a WT and mutants.  

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
# Inference Instructions

Once the model is trained and saved, you can apply it to sequences from a FASTA file to get predicted neq class and attention weights. The results will be saved in your desired path as a JSON file.

```bash 
python get_attn.py --checkpoint best_model.pth --fasta_file your_sequences.fasta (optional)--ss_csv your_ss.csv --ouput path_to_the_final_output
```

If a CSV file for secondary structure prediction is given, the final JSON will also include ss_pred. Such CSV file can be obtained from: https://services.healthtech.dtu.dk/services/NetSurfP-3.0/. This is only useful if you later want to plot attention heatmaps with ss annotations.

Once you run inference, you can visualize the attentin mechanism using `pheatmap_functions.py`
