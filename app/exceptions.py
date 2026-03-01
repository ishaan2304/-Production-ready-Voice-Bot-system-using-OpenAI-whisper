"""
Custom exceptions for VoiceBot system.
"""


class VoiceBotError(Exception):
    """Base exception for all VoiceBot errors."""
    status_code: int = 500

    def __init__(self, message: str, details: str = ""):
        self.message = message
        self.details = details
        super().__init__(message)


class ASRError(VoiceBotError):
    """Raised when speech recognition fails."""
    status_code = 422

    def __init__(self, message: str = "Speech recognition failed", details: str = ""):
        super().__init__(message, details)


class AudioInputError(VoiceBotError):
    """Raised when audio input is invalid or unreadable."""
    status_code = 400

    def __init__(self, message: str = "Invalid audio input", details: str = ""):
        super().__init__(message, details)


class IntentClassificationError(VoiceBotError):
    """Raised when intent classification fails."""
    status_code = 422

    def __init__(self, message: str = "Intent classification failed", details: str = ""):
        super().__init__(message, details)


class ResponseGenerationError(VoiceBotError):
    """Raised when response generation fails."""
    status_code = 500

    def __init__(self, message: str = "Response generation failed", details: str = ""):
        super().__init__(message, details)


class TTSError(VoiceBotError):
    """Raised when text-to-speech synthesis fails."""
    status_code = 500

    def __init__(self, message: str = "Text-to-speech synthesis failed", details: str = ""):
        super().__init__(message, details)


class ModelLoadError(VoiceBotError):
    """Raised when a model fails to load."""
    status_code = 503

    def __init__(self, message: str = "Model loading failed", details: str = ""):
        super().__init__(message, details)


class LowConfidenceError(VoiceBotError):
    """Raised when prediction confidence is below threshold."""
    status_code = 422

    def __init__(self, confidence: float, threshold: float):
        message = f"Confidence {confidence:.2f} below threshold {threshold:.2f}"
        super().__init__(message, "Low confidence prediction")
        self.confidence = confidence
        self.threshold = threshold
