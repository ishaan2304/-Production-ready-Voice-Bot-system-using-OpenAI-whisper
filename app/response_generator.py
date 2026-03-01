"""
Response generation module.
Maps predicted intents to contextually appropriate responses.
Uses template-based generation with random selection for variety.
"""
import random
import time
from typing import Dict, List, Optional, Any

from app.config import get_config
from app.exceptions import ResponseGenerationError
from app.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerator:
    """
    Generates customer support responses based on classified intent.
    Uses template-based mapping with intelligent selection.
    Stays strictly within customer-support scope.
    """

    def __init__(self):
        self.config = get_config()
        self._templates: Dict[str, Any] = {}
        self._loaded = False

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            self._templates = self.config.response_templates.get("intents", {})
            logger.info(f"Response templates loaded: {len(self._templates)} intents")
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load response templates: {e}")
            raise ResponseGenerationError("Failed to load response templates", str(e))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        intent: str,
        confidence: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response for the given intent.

        Args:
            intent: Classified intent label
            confidence: Confidence score for the intent
            context: Optional conversation context

        Returns:
            Dict with response_text, follow_up, intent_used
        """
        start_time = time.perf_counter()
        self._lazy_load()

        try:
            threshold = self.config.response.get("confidence_threshold", 0.4)

            # Use fallback for low-confidence or unknown intents
            if confidence < threshold or intent not in self._templates:
                intent = "fallback"
                logger.debug(f"Using fallback response (confidence={confidence:.3f})")

            template = self._templates.get(intent, self._templates.get("fallback", {}))
            responses: List[str] = template.get("responses", [])
            follow_up: Optional[str] = template.get("follow_up")

            if not responses:
                response_text = self.config.response.get(
                    "fallback_message",
                    "I'm sorry, I couldn't understand your request. Please try again."
                )
            else:
                # Select response — use context hash for determinism if context provided
                if context and context.get("seed"):
                    idx = context["seed"] % len(responses)
                else:
                    idx = random.randint(0, len(responses) - 1)
                response_text = responses[idx]

            elapsed = (time.perf_counter() - start_time) * 1000

            logger.debug(f"Response generated for intent '{intent}' in {elapsed:.1f}ms")

            return {
                "response_text": response_text,
                "intent_used": intent,
                "follow_up": follow_up,
                "processing_time_ms": round(elapsed, 2),
            }

        except ResponseGenerationError:
            raise
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            raise ResponseGenerationError("Response generation failed", str(e))

    def generate_from_text(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: classify intent from text, then generate response.

        Args:
            text: User input text

        Returns:
            Response dict with text, intent, confidence
        """
        from app.intent_classifier import IntentClassifier

        classifier = IntentClassifier()
        intent_result = classifier.predict(text)
        top_intent = intent_result["top_intent"]

        response = self.generate(
            intent=top_intent["intent"],
            confidence=top_intent["confidence"],
            context=context,
        )
        response["intent_confidence"] = top_intent["confidence"]
        response["all_intents"] = intent_result["all_intents"]
        return response

    def list_intents(self) -> List[str]:
        """Return list of all supported intents."""
        self._lazy_load()
        return list(self._templates.keys())

    def is_in_scope(self, text: str) -> bool:
        """
        Check if text appears to be a customer support query.
        Prevents hallucination / off-topic responses.
        """
        scope_keywords = self.config.response.get("scope_keywords", [])
        text_lower = text.lower()
        # Broad check — most text will be in scope
        return any(kw in text_lower for kw in scope_keywords) or len(text.split()) >= 3
