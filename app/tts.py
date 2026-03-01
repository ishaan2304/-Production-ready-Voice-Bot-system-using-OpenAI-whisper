"""
Text-to-Speech (TTS) module.
Converts response text to audio using gTTS or pyttsx3.
Returns audio as bytes.
"""
import io
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from app.config import get_config
from app.exceptions import TTSError
from app.logger import get_logger

logger = get_logger(__name__)


class TTSModule:
    """
    Text-to-speech synthesis module.
    Supports gTTS (online) and pyttsx3 (offline) engines.
    Returns audio as MP3 bytes.
    """

    def __init__(self):
        self.config = get_config().tts
        self._engine = self.config.get("engine", "gtts")
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _check_gtts(self) -> bool:
        """Check if gTTS is available."""
        try:
            import gtts
            return True
        except ImportError:
            return False

    def _check_pyttsx3(self) -> bool:
        """Check if pyttsx3 is available."""
        try:
            import pyttsx3
            return True
        except ImportError:
            return False

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        slow: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Convert text to speech audio.

        Args:
            text: Text to synthesize
            language: Language code (default from config)
            slow: Use slow speech rate (default from config)

        Returns:
            Dict with audio_bytes (MP3), format, duration_estimate
        """
        if not text or not text.strip():
            raise TTSError("Empty text provided for synthesis")

        start_time = time.perf_counter()
        lang = language or self.config.get("language", "en")
        use_slow = slow if slow is not None else self.config.get("slow", False)

        # Clean text for TTS
        clean_text = self._clean_text(text)

        try:
            audio_bytes = self._synthesize_gtts(clean_text, lang, use_slow)
            elapsed = (time.perf_counter() - start_time) * 1000

            # Rough duration estimate: ~150 words/minute
            word_count = len(clean_text.split())
            duration_estimate = (word_count / 150) * 60  # seconds

            logger.debug(f"TTS synthesized {len(clean_text)} chars in {elapsed:.1f}ms")
            self._loaded = True

            return {
                "audio_bytes": audio_bytes,
                "format": "mp3",
                "language": lang,
                "text_length": len(clean_text),
                "duration_estimate_seconds": round(duration_estimate, 1),
                "processing_time_ms": round(elapsed, 2),
            }

        except TTSError:
            raise
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            raise TTSError("Speech synthesis failed", str(e))

    def _synthesize_gtts(self, text: str, language: str, slow: bool) -> bytes:
        """Synthesize using Google Text-to-Speech (gTTS)."""
        try:
            from gtts import gTTS
        except ImportError:
            raise TTSError(
                "gTTS not installed",
                "Install with: pip install gtts"
            )

        try:
            tts = gTTS(text=text, lang=language, slow=slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            # gTTS may fail offline — try pyttsx3 fallback
            logger.warning(f"gTTS failed: {e}. Trying pyttsx3 fallback.")
            return self._synthesize_pyttsx3(text, slow)

    def _synthesize_pyttsx3(self, text: str, slow: bool) -> bytes:
        """Synthesize using pyttsx3 (offline fallback)."""
        try:
            import pyttsx3
        except ImportError:
            raise TTSError(
                "Neither gTTS nor pyttsx3 is available",
                "Install gTTS: pip install gtts OR pyttsx3: pip install pyttsx3"
            )

        try:
            engine = pyttsx3.init()
            rate = engine.getProperty("rate")
            engine.setProperty("rate", rate * 0.8 if slow else rate)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            Path(tmp_path).unlink(missing_ok=True)
            return audio_bytes

        except Exception as e:
            raise TTSError("pyttsx3 synthesis failed", str(e))

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for TTS."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Remove markdown artifacts
        text = text.replace("**", "").replace("*", "").replace("_", "")
        # Limit length
        if len(text) > 2000:
            text = text[:2000] + "..."
        return text.strip()

    def generate_silence(self, duration_ms: int = 500) -> bytes:
        """Generate a short silence MP3 for testing."""
        # Minimal valid MP3 frame (silence)
        # This is a real silent MP3 frame for basic use
        silence_mp3 = bytes([
            0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        return silence_mp3
