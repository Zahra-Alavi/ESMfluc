from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from data_utils import load_and_preprocess_data, create_classification_func
from datasets import Dataset
from evaluate import load
import numpy as np
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Using the ESM for token classification for preidicting the residue of amino acids"
    )
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D", choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"], help="The ESM model to use for token classification. default=esm2_t12_35M_UR50D" )
    
    parser.add_argument("--train_data_file", type=str, default="../../data/train_data.csv", help="Path to the training data CSV file. default=../../data/train_data.csv")
    parser.add_argument("--test_data_file", type=str, default="../../data/test_data.csv", help="Path to the test data CSV file. default=../../data/test_data.csv")
    
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Total number of classes for Neq classification.default=4")
    parser.add_argument("--neq_thresholds", nargs="*", type=float, default=[1.0, 2.0, 4.0],
                        help=("Thresholds for dividing Neq values."
                              "If num_classes=4, you might use 1.0 2.0 4.0. "
                              "For num_classes=3, maybe 1.0 1.5, etc." "default=[1.0, 2.0, 4.0]"))
    
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size. default=2")
    return parser.parse_args()

def compute_metrics(eval_pred):
    metric = load("accuracy")
    predictions, labels = eval_pred
    labels = labels.reshape((-1,))
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))
    predictions = predictions[labels!=-100]
    labels = labels[labels!=-100]
    return metric.compute(predictions=predictions, references=labels)
    
def run(args):
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.esm_model}")
    
    # =========================================================
    # Data Loading and Preprocessing
    # =========================================================
    
    # Load data
    labeled_neq = create_classification_func(args.num_classes, args.neq_thresholds)
    train_data = load_and_preprocess_data(args.train_data_file, labeled_neq)
    test_data = load_and_preprocess_data(args.test_data_file, labeled_neq)

    # Preprocessing data
    X_train = train_data["sequence"].tolist()
    X_test = test_data["sequence"].tolist()
    y_label_train = train_data["neq_class"].tolist()
    y_label_test= test_data["neq_class"].tolist()
    
    X_train_tokenized = tokenizer(X_train)
    X_test_tokenized = tokenizer(X_test)
    
    train_dataset = Dataset.from_dict(X_train_tokenized)
    test_dataset = Dataset.from_dict(X_test_tokenized)
    
    train_dataset = train_dataset.add_column("labels", y_label_train)
    test_dataset = test_dataset.add_column("labels", y_label_test)
    
    model = AutoModelForTokenClassification.from_pretrained(f"facebook/{args.esm_model}", num_labels=args.num_classes)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # =============================================================
    # Training
    # =============================================================
    fp16_state = False
    if torch.cuda.is_available():
        fp16_state = True
        
    training_args = TrainingArguments(
        f"{args.esm_model}-predicting-neq-values-of-residues",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=10,
        weight_decay=0.001,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=fp16_state
    )
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()
    