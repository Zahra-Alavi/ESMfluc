Modified the pipeline to save the attention weights from the self attention layer of BiLSTM, and save the best model for inference. Additionally this new verson allows for DataParallel wrapping so a heavier embedding model can be used with bigger batch size. Example use: 

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



After you have your tained model saved, you can apply it on sequences in a FASTA file and plot the attention heatmap for each entry using attention_visualizer.py:

```bash python attention_visualizer_interactive.py --checkpoint best_model_shuffled.pth --fasta_file examples.fasta ```
