# Regression Implementation Plan for ESMfluc Pipeline

## Current Pipeline Analysis

### What Currently Exists (Classification Only)
1. **Models** ([models.py](models.py)):
   - `BiLSTMClassificationModel` - outputs `num_classes` logits
   - `BiLSTMWithSelfAttentionModel` - outputs `num_classes` logits
   - `TransformerClassificationModel` - outputs `num_classes` logits
   - `ESMLinearTokenClassifier` - outputs `num_classes` logits
   - All use `nn.Linear(hidden_dim, num_classes)` final layer

2. **Loss Functions** ([models.py](models.py)):
   - `FocalLoss` - classification loss with class weighting
   - `CrossEntropyLoss` - standard classification loss

3. **Training** ([train.py](train.py)):
   - Uses `softmax + argmax` for predictions
   - Computes classification metrics (accuracy, precision, recall, F1)
   - Uses `classification_report` and `confusion_matrix`
   - Loss computed on flattened logits: `logits.reshape(-1, num_classes)`

4. **Data Processing** ([data_utils.py](data_utils.py)):
   - `create_classification_func()` - bins continuous Neq values into classes
   - Uses thresholds like [1.0, 2.0, 4.0] to create 4 classes
   - Labels are integers: 0, 1, 2, 3

5. **Arguments** ([arguments.py](arguments.py)):
   - `--num_classes` (default=4)
   - `--neq_thresholds` (default=[1.0, 2.0, 4.0])
   - `--loss_function` (choices: focal, crossentropy)

---

## Recommended Implementation Strategy

### Option 1: Add Regression Models Alongside Classification (RECOMMENDED)

**Why this is best:**
- Preserves existing classification functionality
- Clear separation of concerns
- Easy to switch between modes
- Can run both on same data for comparison

**What to add:**

#### 1. New Regression Models in `models.py`
```python
class BiLSTMRegressionModel(nn.Module):
    """BiLSTM for per-residue regression"""
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_outputs=1, dropout=0.3, bidirectional=1):
        # Same as classification but:
        # - num_outputs instead of num_classes (typically 1)
        # - No final activation (linear output)
        
class BiLSTMWithSelfAttentionRegressionModel(nn.Module):
    """BiLSTM with attention for regression"""
    # Similar structure, regression output

class TransformerRegressionModel(nn.Module):
    """Transformer for regression"""
    # Similar structure, regression output

class ESMLinearTokenRegressor(nn.Module):
    """ESM + linear layer for regression"""
    # Simplest regression model
```

#### 2. New Loss Functions in `models.py`
```python
class MSELoss(nn.Module):
    """Mean Squared Error for regression"""
    
class HuberLoss(nn.Module):
    """Huber loss - robust to outliers"""
    
class WeightedMSELoss(nn.Module):
    """MSE with per-sample weights for imbalanced data"""
```

#### 3. New Arguments in `arguments.py`
```python
--task_type: choices=["classification", "regression"]
--num_outputs: int, default=1 (for regression, usually 1)
--regression_loss: choices=["mse", "mae", "huber", "weighted_mse"]
```

#### 4. Modified Training Functions in `train.py`
```python
def evaluate_regression(model, data_loader, criterion, args):
    """Compute MSE, MAE, R², Pearson correlation"""
    
def set_up_regression_model(args):
    """Create regression model based on architecture"""
    
def get_regression_loss_fn(args, train_dataset):
    """Return appropriate regression loss"""
```

#### 5. Modified Data Utils in `data_utils.py`
```python
class SequenceRegressionDataset(Dataset):
    """Dataset that keeps continuous Neq values"""
    # No binning into classes
    # Returns float labels instead of int
```

---

## Detailed Code Changes

### 1. Add to `models.py`

```python
# ========== REGRESSION MODELS ==========

class BiLSTMRegressionModel(nn.Module):
    """BiLSTM for per-residue regression (e.g., predicting Neq values)"""
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_outputs=1, dropout=0.3, bidirectional=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(self.output_dim, num_outputs)  # Output continuous values

    def forward(self, input_ids, attention_mask, return_features="none"):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)
        h = self.dropout(h)
        output = self.fc(h)  # [B, L, num_outputs]
        
        feats = h if return_features == "pre" else (output if return_features == "post" else None)
        return output, feats


class BiLSTMWithSelfAttentionRegressionModel(nn.Module):
    """BiLSTM with self-attention for regression"""
    def __init__(self, embedding_model, hidden_size, num_layers,
                 num_outputs=1, dropout=0.3, bidirectional=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_model.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout
        )
        self.output_dim = hidden_size * (2 if self.bidirectional else 1)
        self.attention = SelfAttentionLayer(self.output_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.output_dim, num_outputs)

    def forward(self, input_ids, attention_mask, return_attention=False, return_features="none"):
        emb = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(emb)
        ctx, attn = (self.attention(h, attention_mask, True) if return_attention
                     else (self.attention(h, attention_mask), None))
        ctx = self.dropout(ctx)
        output = self.fc(ctx)  # [B, L, num_outputs]
        
        feats = ctx if return_features == "pre" else (output if return_features == "post" else None)
        return (output, feats, attn) if return_attention else (output, feats)


class ESMLinearTokenRegressor(nn.Module):
    """ESM embeddings + linear layer for regression"""
    def __init__(self, embedding_model, num_outputs=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.output_dim = embedding_model.config.hidden_size
        self.fc = nn.Linear(embedding_model.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask, return_features="none", return_attn=False):
        outputs = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attn,
            return_dict=True,
        )
        h = outputs.last_hidden_state
        output = self.fc(h)  # [B, L, num_outputs]

        feats = h if return_features == "pre" else (output if return_features == "post" else None)
        
        if return_attn:
            return output, feats, outputs.attentions
        return output, feats


# ========== REGRESSION LOSS FUNCTIONS ==========

class WeightedMSELoss(nn.Module):
    """MSE loss with optional per-residue weighting and padding mask"""
    def __init__(self, reduction='mean', ignore_value=-100.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_value = ignore_value
    
    def forward(self, predictions, targets, weights=None):
        """
        predictions: [B, L, num_outputs]
        targets: [B, L, num_outputs] or [B, L]
        weights: optional [B, L] weighting mask
        """
        # Ensure same shape
        if targets.dim() == 2 and predictions.dim() == 3:
            targets = targets.unsqueeze(-1)
        
        # Mask out padding (where target == ignore_value)
        mask = (targets != self.ignore_value).float()
        
        # Compute squared error
        sq_error = (predictions - targets) ** 2
        
        # Apply mask
        sq_error = sq_error * mask
        
        # Apply optional weights
        if weights is not None:
            if weights.dim() == 2 and sq_error.dim() == 3:
                weights = weights.unsqueeze(-1)
            sq_error = sq_error * weights
        
        if self.reduction == 'mean':
            return sq_error.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return sq_error.sum()
        else:
            return sq_error
```

### 2. Add to `arguments.py`

```python
# Task type
parser.add_argument("--task_type", type=str, default="classification",
                    choices=["classification", "regression"],
                    help="Task type: classification or regression. default=classification")

# Regression specific
parser.add_argument("--num_outputs", type=int, default=1,
                    help="Number of regression outputs per residue. default=1")

parser.add_argument("--regression_loss", type=str, default="mse",
                    choices=["mse", "mae", "huber", "weighted_mse"],
                    help="Loss function for regression tasks. default=mse")

parser.add_argument("--huber_delta", type=float, default=1.0,
                    help="Delta parameter for Huber loss. default=1.0")
```

### 3. Add to `train.py`

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def set_up_regression_model(args):
    """Create regression model based on architecture"""
    embedding_model = set_up_embedding_model(args)
    
    if args.architecture == "bilstm":
        model = BiLSTMRegressionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_outputs=args.num_outputs,
            bidirectional=args.bidirectional
        )
    elif args.architecture == "bilstm_attention":
        model = BiLSTMWithSelfAttentionRegressionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_outputs=args.num_outputs,
            bidirectional=args.bidirectional
        )
    elif args.architecture == "esm_linear":
        model = ESMLinearTokenRegressor(
            embedding_model=embedding_model,
            num_outputs=args.num_outputs
        )
    else:
        raise ValueError(f"Architecture {args.architecture} not supported for regression")
    
    model.to(args.device)
    
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model


def get_regression_loss_fn(args):
    """Get loss function for regression"""
    if args.regression_loss == "mse":
        return WeightedMSELoss(ignore_value=-100.0)
    elif args.regression_loss == "mae":
        return nn.L1Loss(reduction='none')  # Handle masking manually
    elif args.regression_loss == "huber":
        return nn.HuberLoss(delta=args.huber_delta, reduction='none')
    else:
        raise ValueError(f"Unknown regression loss: {args.regression_loss}")


def evaluate_regression(model, data_loader, criterion, args):
    """Evaluate regression model"""
    m = model.module if isinstance(model, nn.DataParallel) else model
    
    with torch.no_grad():
        m.eval()
        all_preds, all_targets = [], []
        
        for batch in data_loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            y = batch['labels'].to(args.device)  # [B, L] continuous values
            
            output, feats = m(input_ids, attention_mask, return_features="pre")
            output = output.squeeze(-1)  # [B, L, 1] -> [B, L]
            
            # Flatten and filter padding
            output_flat = output.reshape(-1)
            y_flat = y.reshape(-1)
            mask = y_flat != -100.0
            
            all_preds.extend(output_flat[mask].cpu().numpy())
            all_targets.extend(y_flat[mask].cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        pearson_r, pearson_p = pearsonr(all_targets, all_preds)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p
        }
        
        return metrics


def train_regression(args):
    """Training loop for regression"""
    # Similar to train() but:
    # 1. Load continuous Neq values (no classification)
    # 2. Use regression models
    # 3. Use regression loss
    # 4. Use regression metrics
    pass  # Implement similar to train()
```

### 4. Add to `data_utils.py`

```python
class SequenceRegressionDataset(Dataset):
    """Dataset for regression tasks (continuous labels)"""
    def __init__(self, tokenized_sequences, labels):
        """
        tokenized_sequences: list of tokenizer outputs
        labels: list of lists (per-residue continuous values)
        """
        self.tokenized_sequences = tokenized_sequences
        self.labels = labels  # Keep as floats, not classes
    
    def __len__(self):
        return len(self.tokenized_sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_sequences[idx]['input_ids'].squeeze(0),
            'attention_mask': self.tokenized_sequences[idx]['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_regression_data(csv_file):
    """Load data for regression (keep continuous values)"""
    df = pd.read_csv(csv_file)
    # Don't bin Neq values, keep them continuous
    return df
```

---

## Usage Examples

### Classification (current):
```bash
python main.py \
    --task_type classification \
    --architecture bilstm_attention \
    --num_classes 4 \
    --neq_thresholds 1.0 2.0 4.0 \
    --loss_function focal
```

### Regression (new):
```bash
python main.py \
    --task_type regression \
    --architecture bilstm_attention \
    --num_outputs 1 \
    --regression_loss weighted_mse
```

---

## Testing Strategy

1. **Start simple**: Test `ESMLinearTokenRegressor` first
2. **Verify data pipeline**: Ensure continuous labels load correctly
3. **Test loss functions**: MSE should decrease during training
4. **Compare to classification**: Same data, both tasks
5. **Validate metrics**: R² should be reasonable (>0.5 for good model)

---

## Key Differences: Classification vs Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Output layer** | `nn.Linear(hidden, num_classes)` | `nn.Linear(hidden, num_outputs)` |
| **Activation** | Softmax | None (linear) |
| **Loss** | CrossEntropy, Focal | MSE, MAE, Huber |
| **Labels** | Integer classes [0, 1, 2, 3] | Float values [0.5, 1.2, 3.8, ...] |
| **Predictions** | `argmax(softmax(logits))` | Direct output values |
| **Metrics** | Accuracy, F1, Precision | MSE, R², Pearson r |
| **Data prep** | Bin continuous → classes | Keep continuous values |

---

## Next Steps

1. ✅ Review this plan
2. ⬜ Implement regression models in `models.py`
3. ⬜ Add regression arguments to `arguments.py`
4. ⬜ Add regression data loading to `data_utils.py`
5. ⬜ Add regression training functions to `train.py`
6. ⬜ Modify `main.py` to route based on `--task_type`
7. ⬜ Test on small dataset
8. ⬜ Compare classification vs regression performance
