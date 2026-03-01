"""
Intent classification model training script.
Fine-tunes DistilBERT on customer support intents.
"""
import json
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Intent label mapping
INTENT_LABELS = [
    "order_status",
    "order_cancellation",
    "refund_request",
    "subscription_inquiry",
    "account_issues",
    "payment_issues",
    "shipping_inquiry",
    "return_request",
    "technical_support",
    "product_inquiry",
]

LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(INTENT_LABELS)}


def load_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    """Load intent dataset from JSON file."""
    with open(data_path, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [LABEL2ID[item["intent"]] for item in data]
    return texts, labels


def split_dataset(
    texts: List[str],
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple:
    """Split dataset into train/val/test."""
    random.seed(seed)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        list(texts[:train_end]), list(labels[:train_end]),
        list(texts[train_end:val_end]), list(labels[train_end:val_end]),
        list(texts[val_end:]), list(labels[val_end:]),
    )


def train_model(
    data_path: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Dict:
    """
    Fine-tune DistilBERT for intent classification.
    Returns evaluation metrics.
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from torch.utils.data import Dataset
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix
    )

    print(f"\n{'='*60}")
    print(f"Training Intent Classifier")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load data
    texts, labels = load_dataset(data_path)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_dataset(
        texts, labels, seed=seed
    )

    print(f"Dataset split: {len(train_texts)} train / {len(val_texts)} val / {len(test_texts)} test")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class IntentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)
    test_dataset = IntentDataset(test_texts, test_labels, tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        seed=seed,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),      # ← changed to this
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,                 # ← changed to this
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_metrics = compute_metrics(
        (test_results.predictions, test_results.label_ids)
    )

    # Confusion matrix
    test_preds = np.argmax(test_results.predictions, axis=-1)
    cm = confusion_matrix(test_results.label_ids, test_preds)

    print(f"\n{'='*60}")
    print("TEST SET METRICS:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"{'='*60}\n")

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "fine_tuned": True,
        "num_labels": len(INTENT_LABELS),
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "test_metrics": test_metrics,
        "training_time_seconds": training_time,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save confusion matrix data
    cm_data = {
        "matrix": cm.tolist(),
        "labels": INTENT_LABELS,
    }
    with open(output_path / "confusion_matrix.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    print(f"Model saved to: {output_dir}")
    return {
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "labels": INTENT_LABELS,
        "training_time": training_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--data", default="data/intent_dataset.json")
    parser.add_argument("--output", default="models/intent_classifier")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
