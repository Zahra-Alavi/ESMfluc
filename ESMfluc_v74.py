#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:01:42 2024

@author: zalavi
"""



import logging
from transformers import EsmTokenizer, EsmModel
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import ast
import seaborn as sns
import random
from sklearn.model_selection import KFold
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Load and process training data
# =============================================================================

# Load training data
training_data = pd.read_csv('neq_training_data.csv')
rows, columns = training_data.shape

# Convert the 'neq' column from strings to lists
training_data['neq'] = training_data['neq'].apply(ast.literal_eval)

# Check for mismatches between sequence lengths and Neq lists
for idx, row in training_data.iterrows():
    sequence_length = len(row['sequence'])
    neq_length = len(row['neq'])
    if sequence_length != neq_length:
        print(f"Mismatch at index {idx}: sequence length is {sequence_length}, but neq length is {neq_length}")

# Check for empty sequences and Neq values
training_data = training_data.dropna(subset=['sequence', 'neq'])
training_data = training_data[
    training_data['sequence'].str.len() == training_data['neq'].apply(len)
].reset_index(drop=True)

# Extract Neq values 
neq_values = [value for neq in training_data['neq'] for value in neq]

# Plot probability density of Neq values 
plt.figure(figsize=(10, 6))
sns.kdeplot(neq_values, bw_adjust=0.5)
plt.xlabel('Neq Values')
plt.ylabel('Density')
plt.title('Probability Density of Neq Values Before Normalization')
plt.show()


# Function to plot Neq vs. Residue number for random samples
def plot_neq_vs_residue(training_data, num_samples=3):
    sample_indices = random.sample(range(len(training_data)), num_samples)
    sample_sequences = training_data.iloc[sample_indices]

    plt.figure(figsize=(15, 10))
    for idx, row in sample_sequences.iterrows():
        sequence = row['sequence']
        neq_values = row['neq']
        residue_numbers = list(range(1, len(neq_values) + 1))

        plt.plot(residue_numbers, neq_values, label=f'Sample {idx}')

    plt.xlabel('Residue Number')
    plt.ylabel('Neq')
    plt.title('Neq vs. Residue Number for 10 Random Samples')
    plt.legend(loc='upper right')
    plt.show()

# Call the function to plot RMSF vs. Residue number for 10 random samples
plot_neq_vs_residue(training_data, num_samples=3)


# Flatten all Neq values into a single list
all_neq_values = [neq for sublist in training_data['neq'] for neq in sublist]
neq_series = pd.Series(all_neq_values)

# Step 2: Define bins and labels
bin_edges = [0.9999999, 1.0000001] + [float(i) for i in range(2, 17)]
labels = ['Exactly 1.0'] + [f'Between {i} and {i+1}' for i in range(1, 16)]

# Step 3: Bin the Neq values and count frequencies
neq_binned = pd.cut(
    neq_series,
    bins=bin_edges,
    labels=labels,
    right=True,
    include_lowest=True
)
frequency_counts = neq_binned.value_counts().sort_index()

print("Frequency counts of Neq values:")
print(frequency_counts)

# Step 4: Plot the bar chart
frequency_counts.plot(kind='bar', figsize=(12,6), color='skyblue', edgecolor='black')

plt.xlabel('Neq Value Ranges')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Neq Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# =============================================================================
# Classify Neq values
# =============================================================================

def classify_neq(neq_value):
    if neq_value == 1.0:
        return 0  # Class 0 (Very Rigid)
# =============================================================================
#     elif 1.0 < neq_value <= 1.5:
#         return 1  # Class 1 (Rigid)
# =============================================================================
    else:  # Neq > 4.0
        return 1  # Class 1 (Flexible)

# Apply classification to each residue's Neq value
training_data['neq_class'] = training_data['neq'].apply(
    lambda neq_list: [classify_neq(neq) for neq in neq_list]
)

# Flatten the list of neq_class to get all classifications in one list
all_neq_classes = [class_value for class_list in training_data['neq_class'] for class_value in class_list]

# Count the occurrences of each class
class_counts = pd.Series(all_neq_classes).value_counts().sort_index()

# Print the population of each class
print(f"Class 0 (Very Rigid): {class_counts[0]}")
print(f"Class 1 (Rigid): {class_counts[1]}")


# =============================================================================
# Load tokenizer
# =============================================================================

# Initialize tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

# Tokenize sequences
encoded_inputs = [
    tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False)
    for seq in training_data['sequence']
]

# =============================================================================
# Create Custom Dataset and DataLoader
# =============================================================================

class SequenceClassificationDataset(Dataset):
    def __init__(self, encoded_inputs, labels):
        self.encoded_inputs = encoded_inputs  # List of dictionaries with 'input_ids' and 'attention_mask'
        self.labels = labels                  # List of lists [sequence_length]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encoded_inputs[idx]['input_ids'].squeeze(0),          # [sequence_length]
            'attention_mask': self.encoded_inputs[idx]['attention_mask'].squeeze(0),# [sequence_length]
            'labels': torch.tensor(self.labels[idx])                                # [sequence_length]
        }

# Prepare labels
sequence_labels = training_data['neq_class'].tolist()  # List of lists [sequence_length]

# Create the classification dataset
classification_dataset = SequenceClassificationDataset(encoded_inputs, sequence_labels)

# Define collate function
def collate_fn_sequence(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return {
        'input_ids': input_ids_padded,  # [batch_size, max_seq_len]
        'attention_mask': attention_masks_padded,  # [batch_size, max_seq_len]
        'labels': labels_padded  # [batch_size, max_seq_len]
    }

# Create data loaders
batch_size = 4  # Adjust based on your GPU memory
train_loader = DataLoader(classification_dataset, batch_size=batch_size, collate_fn=collate_fn_sequence, shuffle=True)

# =============================================================================
# Define Focal Loss with custom alpha
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        logpt = -self.ce_loss(inputs, targets)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        else:
            return -loss

# =============================================================================
# # Define custom alpha values for Focal Loss
# alpha = torch.tensor([1, 1.5, 2.0], dtype=torch.float).to('cuda')
# =============================================================================
loss_fn = FocalLoss(alpha=None, gamma=2, ignore_index=-1)

# =============================================================================
# Define Evaluation Functions
# =============================================================================

def compute_validation_loss(model, data_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            outputs_flat = outputs.view(-1, num_classes)
            labels_flat = labels.view(-1)

            loss = loss_fn(outputs_flat, labels_flat)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss

def evaluate_model_and_save_plots(model, data_loader, device, fold):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)  # [batch_size, seq_len, num_classes]
            probs = torch.softmax(outputs, dim=2)
            preds = torch.argmax(probs, dim=2)

            # Flatten tensors
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            mask_flat = (labels_flat != -1)

            # Apply mask to filter out padding positions
            valid_preds = preds_flat[mask_flat]
            valid_labels = labels_flat[mask_flat]

            all_preds.extend(valid_preds.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    report = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Fold {fold} Classification Report:\n{report}")
    print(f"Fold {fold} Confusion Matrix:\n{cm}")

    # Save classification report to a text file
    with open(f'classification_report_fold_{fold}.txt', 'w') as f:
        f.write(f"Fold {fold} Classification Report:\n{report}\n")
        f.write(f"Fold {fold} Confusion Matrix:\n{cm}\n")

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(f'confusion_matrix_fold_{fold}.png')
    plt.close()

    return report, cm
# =============================================================================
# Define Attention Layer and BiLSTM-based Classification Model
# =============================================================================

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.key = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.value = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_output):
        # lstm_output is [batch_size, seq_len, hidden_size*2] for bidirectional LSTM
        Q = self.query(lstm_output)  # [batch_size, seq_len, hidden_size*2]
        K = self.key(lstm_output)    # [batch_size, seq_len, hidden_size*2]
        V = self.value(lstm_output)  # [batch_size, seq_len, hidden_size*2]

        # Compute attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_weights = self.softmax(attention_scores)  # Normalize attention weights

        # Weighted sum of values based on attention weights
        context = torch.bmm(attention_weights, V)  # [batch_size, seq_len, hidden_size*2]

        return context


class BiLSTMWithSelfAttentionModel(nn.Module):
    def __init__(self, embedding_model, hidden_size, num_layers, num_classes=2, dropout=0.3):
        super(BiLSTMWithSelfAttentionModel, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attention = SelfAttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Output per amino acid

    def forward(self, input_ids, attention_mask):
        # Generate embeddings using the embedding model
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Pass through BiLSTM
        lstm_out, _ = self.lstm(embeddings)  # [batch_size, seq_len, hidden_size*2]

        # Apply self-attention to get context-aware embeddings
        context_embeddings = self.attention(lstm_out)  # [batch_size, seq_len, hidden_size*2]

        # Apply dropout and predict per amino acid label
        context_embeddings = self.dropout(context_embeddings)
        output = self.fc(context_embeddings)  # [batch_size, seq_len, num_classes]

        return output

# =============================================================================
# Initialize Model, Optimizer, and Scheduler
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

num_classes =2

# Cross-Validation Training Loop
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_ids, test_ids) in enumerate(kfold.split(classification_dataset)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    # Split the data into training and validation sets
    train_sequences = [training_data.iloc[idx]['sequence'] for idx in train_ids]
    train_labels = [training_data.iloc[idx]['neq_class'] for idx in train_ids]
    train_encoded_inputs = [encoded_inputs[idx] for idx in train_ids]

    val_sequences = [training_data.iloc[idx]['sequence'] for idx in test_ids]
    val_labels = [training_data.iloc[idx]['neq_class'] for idx in test_ids]
    val_encoded_inputs = [encoded_inputs[idx] for idx in test_ids]

    # Create training dataset
    train_dataset = SequenceClassificationDataset(train_encoded_inputs, train_labels)
    
    # Create validation dataset
    val_dataset = SequenceClassificationDataset(val_encoded_inputs, val_labels)

    # Define data loaders
    batch_size = 4  # Adjust based on your GPU memory
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn_sequence, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn_sequence
    )


    model_checkpoint = "facebook/esm2_t12_35M_UR50D"
    embedding_model = EsmModel.from_pretrained(model_checkpoint)
    embedding_model.train()
    embedding_model.to(device)

    # Unfreeze more layers for fine-tuning (unfreeze up to layer 8)
    for name, param in embedding_model.named_parameters():
        param.requires_grad = False
        if any(f'encoder.layer.{i}' in name for i in range(5, 12)) or 'layer_norm' in name:
            param.requires_grad = True

    # Initialize the model with 3 LSTM layers and attention
    model = BiLSTMWithSelfAttentionModel(
        embedding_model=embedding_model,
        hidden_size=512,
        num_layers=3,
        num_classes=2,
        dropout=0.3
        )

    model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # =============================================================================
    # Training Loop
    # =============================================================================


    num_epochs = 20
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Mixed precision
                outputs = model(input_ids, attention_mask)
                outputs_flat = outputs.view(-1, num_classes)
                labels_flat = labels.view(-1)
                loss = loss_fn(outputs_flat, labels_flat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}')

        # Compute validation loss for early stopping
   
        avg_val_loss = compute_validation_loss(model, val_loader, loss_fn, device)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                break

    # Plot loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - Fold {fold + 1}')
    plt.legend()
    plt.savefig(f'loss_curves_fold_{fold+1}.png')
    plt.close()
    
    # Load the best model before evaluation
    model.load_state_dict(torch.load(f'best_model_fold_{fold+1}.pt'))

    # Evaluation
    report, cm = evaluate_model_and_save_plots(model, val_loader, device, fold + 1)

print('Training and evaluation complete.')
