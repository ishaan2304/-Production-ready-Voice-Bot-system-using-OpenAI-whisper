"""
Test suite for VoiceBot components.
Tests ASR, intent classification, response generation, and TTS modules.
"""
import io
import json
import wave
import sys
import struct
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_test_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file for testing."""
    n_frames = int(duration_seconds * sample_rate)
    # Generate a 440Hz tone
    t = np.linspace(0, duration_seconds, n_frames, False)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        from app.config import get_config
        cfg = get_config()
        assert cfg.app.get("name") is not None
        assert cfg.asr.get("model_name") is not None
        assert cfg.intent.get("num_labels") == 10

    def test_response_templates_load(self):
        from app.config import get_config
        cfg = get_config()
        templates = cfg.response_templates
        assert "intents" in templates
        assert "order_status" in templates["intents"]
        assert "refund_request" in templates["intents"]

    def test_all_intents_have_responses(self):
        from app.config import get_config
        from app.intent_classifier import INTENT_LABELS
        cfg = get_config()
        templates = cfg.response_templates["intents"]
        for intent in INTENT_LABELS:
            assert intent in templates, f"Missing template for intent: {intent}"
            assert len(templates[intent]["responses"]) > 0


class TestASR:
    """Test ASR module."""

    def test_validate_valid_wav(self):
        from app.asr import ASRModule
        asr = ASRModule()
        wav_bytes = create_test_wav()
        audio_array, sample_rate = asr.validate_audio(wav_bytes)
        assert len(audio_array) > 0
        assert sample_rate == 16000
        assert audio_array.dtype == np.float32
        assert np.max(np.abs(audio_array)) <= 1.0

    def test_reject_empty_audio(self):
        from app.asr import ASRModule
        from app.exceptions import AudioInputError
        asr = ASRModule()
        with pytest.raises(AudioInputError):
            asr.validate_audio(b"")

    def test_reject_invalid_wav(self):
        from app.asr import ASRModule
        from app.exceptions import AudioInputError
        asr = ASRModule()
        with pytest.raises(AudioInputError):
            asr.validate_audio(b"not a wav file at all")

    def test_reject_too_short(self):
        from app.asr import ASRModule
        from app.exceptions import AudioInputError
        asr = ASRModule()
        wav_bytes = create_test_wav(duration_seconds=0.05)
        with pytest.raises(AudioInputError, match="too short"):
            asr.validate_audio(wav_bytes)

    def test_stereo_to_mono_conversion(self):
        from app.asr import ASRModule
        asr = ASRModule()

        n_frames = 16000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            stereo = np.zeros(n_frames * 2, dtype=np.int16)
            wf.writeframes(stereo.tobytes())
        wav_bytes = buf.getvalue()

        audio_array, sr = asr.validate_audio(wav_bytes)
        assert len(audio_array) == n_frames  # Mono

    def test_wer_calculation(self):
        from app.asr import ASRModule
        asr = ASRModule()
        refs = ["hello world", "test sentence"]
        hyps = ["hello world", "test sentence"]
        metrics = asr.compute_wer(refs, hyps)
        assert metrics["wer"] == 0.0

    def test_wer_with_errors(self):
        from app.asr import ASRModule
        asr = ASRModule()
        refs = ["hello world", "the quick brown fox"]
        hyps = ["hello world", "the slow brown fox"]
        metrics = asr.compute_wer(refs, hyps)
        assert 0.0 < metrics["wer"] <= 1.0


class TestIntentClassifier:
    """Test intent classification."""

    def setup_method(self):
        from app.intent_classifier import IntentClassifier
        self.classifier = IntentClassifier()

    def test_classify_order_status(self):
        result = self.classifier.predict("Where is my order?")
        assert result["top_intent"]["intent"] == "order_status"
        assert 0.0 <= result["top_intent"]["confidence"] <= 1.0

    def test_classify_refund_request(self):
        result = self.classifier.predict("I want a refund please")
        assert result["top_intent"]["intent"] == "refund_request"

    def test_classify_cancellation(self):
        result = self.classifier.predict("Please cancel my order immediately")
        assert result["top_intent"]["intent"] == "order_cancellation"

    def test_classify_subscription(self):
        result = self.classifier.predict("How do I cancel my subscription plan?")
        assert result["top_intent"]["intent"] == "subscription_inquiry"

    def test_returns_all_intents(self):
        result = self.classifier.predict("I have a payment problem")
        assert len(result["all_intents"]) == 10
        # Should be sorted by confidence
        confidences = [i["confidence"] for i in result["all_intents"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidence_threshold(self):
        result = self.classifier.predict("I have a question")
        assert "is_confident" in result
        assert isinstance(result["is_confident"], bool)

    def test_rejects_empty_text(self):
        from app.exceptions import IntentClassificationError
        with pytest.raises(IntentClassificationError):
            self.classifier.predict("")

    def test_all_10_intents_have_display_names(self):
        from app.intent_classifier import INTENT_DISPLAY_NAMES, INTENT_LABELS
        for label in INTENT_LABELS:
            assert label in INTENT_DISPLAY_NAMES
            assert INTENT_DISPLAY_NAMES[label] != ""


class TestResponseGenerator:
    """Test response generation."""

    def setup_method(self):
        from app.response_generator import ResponseGenerator
        self.generator = ResponseGenerator()

    def test_generates_response_for_known_intent(self):
        result = self.generator.generate(intent="order_status", confidence=0.9)
        assert len(result["response_text"]) > 10
        assert result["intent_used"] == "order_status"

    def test_fallback_for_low_confidence(self):
        result = self.generator.generate(intent="order_status", confidence=0.1)
        assert result["intent_used"] == "fallback"

    def test_fallback_for_unknown_intent(self):
        result = self.generator.generate(intent="unknown_xyz", confidence=0.9)
        assert result["intent_used"] == "fallback"

    def test_all_intents_have_responses(self):
        from app.intent_classifier import INTENT_LABELS
        for intent in INTENT_LABELS:
            result = self.generator.generate(intent=intent, confidence=0.9)
            assert result["response_text"]
            assert result["intent_used"] == intent

    def test_follow_up_included(self):
        result = self.generator.generate(intent="order_status", confidence=0.9)
        assert "follow_up" in result

    def test_response_is_grammatical(self):
        result = self.generator.generate(intent="refund_request", confidence=0.9)
        text = result["response_text"]
        # Basic checks: starts with capital, ends with punctuation
        assert text[0].isupper()
        assert text[-1] in ".?!"

    def test_list_intents(self):
        intents = self.generator.list_intents()
        assert len(intents) >= 10
        assert "order_status" in intents
        assert "fallback" in intents


class TestTTS:
    """Test text-to-speech module."""

    def setup_method(self):
        from app.tts import TTSModule
        self.tts = TTSModule()

    def test_clean_text(self):
        text = "**Hello** _world_  extra  spaces"
        cleaned = self.tts._clean_text(text)
        assert "**" not in cleaned
        assert "_" not in cleaned
        assert "  " not in cleaned

    def test_truncates_long_text(self):
        long_text = "word " * 500
        cleaned = self.tts._clean_text(long_text)
        assert len(cleaned) <= 2010  # 2000 + "..."


class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "uptime_seconds" in data

    def test_predict_intent(self, client):
        response = client.post(
            "/predict-intent",
            json={"text": "Where is my order?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "top_intent" in data
        assert data["top_intent"]["intent"] == "order_status"
        assert len(data["all_intents"]) == 10
        assert "request_id" in data

    def test_generate_response(self, client):
        response = client.post(
            "/generate-response",
            json={"text": "I want a refund"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response_text" in data
        assert len(data["response_text"]) > 10

    def test_generate_response_with_intent(self, client):
        response = client.post(
            "/generate-response",
            json={"text": "anything", "intent": "shipping_inquiry"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent_used"] == "shipping_inquiry"

    def test_transcribe_rejects_non_wav(self, client):
        response = client.post(
            "/transcribe",
            files={"audio": ("test.mp3", b"fake audio", "audio/mpeg")},
        )
        assert response.status_code == 400

    def test_error_response_format(self, client):
        response = client.post(
            "/predict-intent",
            json={"text": ""},
        )
        assert response.status_code in (400, 422)

    def test_request_id_in_headers(self, client):
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_openapi_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema["paths"]
        assert "/transcribe" in paths
        assert "/predict-intent" in paths
        assert "/generate-response" in paths
        assert "/synthesize" in paths
        assert "/voicebot" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
