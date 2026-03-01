"""
Intent classification module.
Loads fine-tuned DistilBERT model and classifies customer support intents.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from app.config import get_config
from app.exceptions import IntentClassificationError, ModelLoadError
from app.logger import get_logger

logger = get_logger(__name__)

INTENT_DISPLAY_NAMES = {
    "order_status": "Order Status",
    "order_cancellation": "Order Cancellation",
    "refund_request": "Refund Request",
    "subscription_inquiry": "Subscription Inquiry",
    "account_issues": "Account Issues",
    "payment_issues": "Payment Issues",
    "shipping_inquiry": "Shipping Inquiry",
    "return_request": "Return Request",
    "technical_support": "Technical Support",
    "product_inquiry": "Product Inquiry",
    "fallback": "General Inquiry",
}

INTENT_LABELS = list(INTENT_DISPLAY_NAMES.keys())[:-1]  # Exclude fallback


class IntentClassifier:
    """
    Intent classification using fine-tuned DistilBERT.
    Falls back to a keyword-based classifier if no trained model is available.
    """

    def __init__(self):
        self.config = get_config().intent
        self._model = None
        self._tokenizer = None
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}
        self._model_loaded = False
        self._use_fallback = False

    def _lazy_load(self) -> None:
        """Lazily load the intent classification model."""
        if self._model_loaded:
            return

        model_path = self.config.get("fine_tuned_path", "models/intent_classifier")
        full_path = Path(__file__).resolve().parent.parent / model_path

        if full_path.exists() and (full_path / "config.json").exists():
            self._load_fine_tuned(str(full_path))
        else:
            logger.warning(
                f"Fine-tuned model not found at {full_path}. "
                "Using keyword-based fallback classifier. "
                "Run: python scripts/train_intent_classifier.py"
            )
            self._use_fallback = True
            self._model_loaded = True

    def _load_fine_tuned(self, model_path: str) -> None:
        """Load fine-tuned transformer model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading intent classifier from {model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._model.eval()

            # Load metadata
            meta_path = Path(model_path) / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self._label2id = meta.get("label2id", {})
                self._id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}
            else:
                self._id2label = {i: label for i, label in enumerate(INTENT_LABELS)}
                self._label2id = {v: k for k, v in self._id2label.items()}

            self._model_loaded = True
            self._use_fallback = False
            logger.info("Intent classifier loaded successfully (transformer model)")

        except ImportError as e:
            raise ModelLoadError("transformers or torch not installed", str(e))
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model: {e}. Using fallback.")
            self._use_fallback = True
            self._model_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def using_fallback(self) -> bool:
        return self._use_fallback

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify intent from text.

        Args:
            text: User input text

        Returns:
            Dict with top_intent, confidence, all_intents, is_confident
        """
        if not text or not text.strip():
            raise IntentClassificationError("Empty text provided for intent classification")

        start_time = time.perf_counter()
        self._lazy_load()

        try:
            if self._use_fallback:
                result = self._keyword_classify(text)
            else:
                result = self._transformer_classify(text)

            threshold = self.config.get("confidence_threshold", 0.5)
            result["is_confident"] = result["top_intent"]["confidence"] >= threshold
            result["processing_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)

            logger.debug(
                f"Intent classified: '{result['top_intent']['intent']}' "
                f"({result['top_intent']['confidence']:.3f}) in {result['processing_time_ms']}ms"
            )
            return result

        except IntentClassificationError:
            raise
        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)
            raise IntentClassificationError("Classification failed", str(e))

    def _transformer_classify(self, text: str) -> Dict[str, Any]:
        """Classify using fine-tuned transformer model."""
        import torch
        import torch.nn.functional as F

        max_length = self.config.get("max_length", 128)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0).numpy()

        all_intents = [
            {
                "intent": self._id2label.get(i, f"intent_{i}"),
                "confidence": float(probs[i]),
                "display_name": INTENT_DISPLAY_NAMES.get(
                    self._id2label.get(i, ""), f"Intent {i}"
                ),
            }
            for i in range(len(probs))
        ]
        all_intents.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "text": text,
            "top_intent": all_intents[0],
            "all_intents": all_intents,
        }

    def _keyword_classify(self, text: str) -> Dict[str, Any]:
        """Rule-based keyword classifier as fallback."""
        text_lower = text.lower()

        keyword_map = {
            "order_status": [
                "order status", "where is my order", "track", "arrived", "delivered",
                "when will", "shipment status", "package", "late", "delayed",
            ],
            "order_cancellation": [
                "cancel", "cancellation", "don't want", "remove order", "stop order",
                "revoke", "abort order",
            ],
            "refund_request": [
                "refund", "money back", "reimburs", "return payment", "paid too much",
                "overcharged", "charge back", "credit",
            ],
            "subscription_inquiry": [
                "subscription", "subscribe", "plan", "upgrade", "downgrade", "renew",
                "monthly", "annual", "billing cycle", "pause subscription",
            ],
            "account_issues": [
                "account", "login", "password", "locked", "can't access", "profile",
                "username", "sign in", "credentials", "forgot password",
            ],
            "payment_issues": [
                "payment", "charge", "declined", "credit card", "billing",
                "invoice", "charged twice", "fraud", "paypal", "bank",
            ],
            "shipping_inquiry": [
                "shipping", "delivery", "ship", "fedex", "ups", "usps", "courier",
                "address", "express", "standard shipping", "tracking number",
            ],
            "return_request": [
                "return", "send back", "exchange", "wrong item", "damaged",
                "defective", "return label", "return policy",
            ],
            "technical_support": [
                "error", "bug", "crash", "not working", "broken", "issue",
                "problem", "technical", "glitch", "loading", "404", "page",
            ],
            "product_inquiry": [
                "product", "feature", "price", "cost", "available", "stock",
                "compatible", "specification", "warranty", "version", "what is",
            ],
        }

        scores: Dict[str, float] = {intent: 0.0 for intent in INTENT_LABELS}
        for intent, keywords in keyword_map.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] += 1.0 / len(keywords)

        total = sum(scores.values())
        if total == 0:
            # Uniform distribution fallback
            uniform = 1.0 / len(INTENT_LABELS)
            probs = {intent: uniform for intent in INTENT_LABELS}
        else:
            probs = {intent: score / total for intent, score in scores.items()}

        # Softmax-like normalization
        max_score = max(probs.values())
        if max_score < 0.1:
            # Low confidence — boost fallback behavior
            probs = {intent: score * 0.5 + 0.05 for intent, score in probs.items()}

        sorted_intents = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        all_intents = [
            {
                "intent": intent,
                "confidence": round(conf, 4),
                "display_name": INTENT_DISPLAY_NAMES.get(intent, intent),
            }
            for intent, conf in sorted_intents
        ]

        return {
            "text": text,
            "top_intent": all_intents[0],
            "all_intents": all_intents,
        }
