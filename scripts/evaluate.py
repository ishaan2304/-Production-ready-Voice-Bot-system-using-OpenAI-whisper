"""
Evaluation script for VoiceBot components.
Generates metrics reports and confusion matrix visualizations.
"""
import json
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate_intent_classifier(data_path: str, model_path: str = None):
    """Evaluate intent classifier and print metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
    import numpy as np

    print("\n" + "=" * 60)
    print("INTENT CLASSIFIER EVALUATION")
    print("=" * 60)

    # Load data
    with open(data_path) as f:
        data = json.load(f)

    from app.intent_classifier import IntentClassifier, INTENT_LABELS
    classifier = IntentClassifier()

    y_true, y_pred, confidences = [], [], []
    print(f"Evaluating on {len(data)} samples...")

    for item in data:
        result = classifier.predict(item["text"])
        y_true.append(item["intent"])
        y_pred.append(result["top_intent"]["intent"])
        confidences.append(result["top_intent"]["confidence"])

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=INTENT_LABELS)

    print(f"\n{'─'*40}")
    print(f"  Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision (W):    {precision:.4f}")
    print(f"  Recall (W):       {recall:.4f}")
    print(f"  F1-Score (W):     {f1:.4f}")
    print(f"  Avg Confidence:   {sum(confidences)/len(confidences):.4f}")
    print(f"  Classifier Type:  {'Transformer' if not classifier.using_fallback else 'Keyword (Fallback)'}")
    print(f"{'─'*40}")

    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, labels=INTENT_LABELS, zero_division=0))

    # Save confusion matrix visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        short_labels = [l.replace("_", "\n") for l in INTENT_LABELS]
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(
            f"Intent Classifier Confusion Matrix\n"
            f"Accuracy: {acc:.4f} | F1: {f1:.4f} | n={len(data)}",
            fontsize=14,
            pad=15,
        )
        ax.set_ylabel("True Intent", fontsize=12)
        ax.set_xlabel("Predicted Intent", fontsize=12)
        plt.tight_layout()

        output_path = Path("models/confusion_matrix.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nConfusion matrix saved to: {output_path}")

    except ImportError:
        print("matplotlib/seaborn not installed — skipping confusion matrix plot")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_asr_wer(test_set_path: str):
    """Evaluate ASR Word Error Rate on test set."""
    print("\n" + "=" * 60)
    print("ASR WORD ERROR RATE EVALUATION")
    print("=" * 60)

    if not Path(test_set_path).exists():
        print(f"Test set not found: {test_set_path}")
        print("Creating sample test set for demonstration...")

        sample_test_set = [
            {"reference": "where is my order", "hypothesis": "where is my order"},
            {"reference": "i want to cancel my order", "hypothesis": "i want to cancel my order"},
            {"reference": "please process a refund for me", "hypothesis": "please process a refund for me"},
            {"reference": "my payment was declined", "hypothesis": "my payment declined"},
            {"reference": "how do i return this item", "hypothesis": "how do i return this item"},
        ]
        with open(test_set_path, "w") as f:
            json.dump(sample_test_set, f, indent=2)
        print(f"Sample test set created at: {test_set_path}")

    with open(test_set_path) as f:
        test_data = json.load(f)

    from app.asr import ASRModule
    asr = ASRModule()

    references = [item["reference"] for item in test_data]
    hypotheses = [item["hypothesis"] for item in test_data]

    metrics = asr.compute_wer(references, hypotheses)

    print(f"\n  WER:          {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    if metrics.get("cer") is not None:
        print(f"  CER:          {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"  Test Samples: {metrics['num_samples']}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VoiceBot components")
    parser.add_argument("--component", choices=["intent", "asr", "all"], default="all")
    parser.add_argument("--data", default="data/intent_dataset.json")
    parser.add_argument("--asr-test", default="data/asr_test_set.json")
    args = parser.parse_args()

    if args.component in ("intent", "all"):
        evaluate_intent_classifier(args.data)

    if args.component in ("asr", "all"):
        evaluate_asr_wer(args.asr_test)
