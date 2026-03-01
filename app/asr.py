"""
Automatic Speech Recognition (ASR) module.
Uses OpenAI Whisper via HuggingFace transformers for speech-to-text.
"""
import io
import time
import wave
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from app.config import get_config
from app.exceptions import ASRError, AudioInputError, ModelLoadError
from app.logger import get_logger

logger = get_logger(__name__)


class ASRModule:
    """
    Speech recognition module wrapping OpenAI Whisper.
    Handles WAV input, noise tolerance, and confidence reporting.
    """

    def __init__(self):
        self.config = get_config().asr
        self._pipeline = None
        self._model_loaded = False

    def _lazy_load(self) -> None:
        """Lazily load the Whisper model on first use."""
        if self._model_loaded:
            return
        try:
            logger.info(f"Loading ASR model: {self.config.get('model_name', 'openai/whisper-base')}")
            from transformers import pipeline as hf_pipeline
            import torch

            device = self.config.get("device", "cpu")
            model_name = self.config.get("model_name", "openai/whisper-base")

            self._pipeline = hf_pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                device=0 if device == "cuda" else -1,
                chunk_length_s=self.config.get("chunk_length_s", 30),
                return_timestamps=False,
            )
            self._model_loaded = True
            logger.info("ASR model loaded successfully")
        except ImportError as e:
            raise ModelLoadError("transformers or torch not installed", str(e))
        except Exception as e:
            raise ModelLoadError(f"Failed to load ASR model", str(e))

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def validate_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Validate and parse WAV audio bytes.
        Returns (audio_array, sample_rate).
        """
        if not audio_bytes:
            raise AudioInputError("Empty audio data received")

        if len(audio_bytes) < 44:  # WAV header minimum
            raise AudioInputError("Audio data too small to be valid WAV")

        try:
            with io.BytesIO(audio_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()

                    if n_frames == 0:
                        raise AudioInputError("Audio file contains no audio frames")

                    duration = n_frames / framerate
                    if duration < 0.1:
                        raise AudioInputError(f"Audio too short: {duration:.2f}s (minimum 0.1s)")
                    if duration > 300:
                        raise AudioInputError(f"Audio too long: {duration:.2f}s (maximum 300s)")

                    raw = wf.readframes(n_frames)
                    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
                    audio_array = np.frombuffer(raw, dtype=dtype).astype(np.float32)

                    # Normalize to [-1, 1]
                    max_val = np.iinfo(dtype).max
                    audio_array = audio_array / max_val

                    # Convert stereo to mono
                    if n_channels == 2:
                        audio_array = audio_array.reshape(-1, 2).mean(axis=1)

                    logger.debug(
                        f"Audio validated: {duration:.2f}s, {framerate}Hz, "
                        f"{n_channels}ch, {sampwidth*8}bit"
                    )
                    return audio_array, framerate

        except AudioInputError:
            raise
        except wave.Error as e:
            raise AudioInputError("Invalid WAV format", str(e))
        except Exception as e:
            raise AudioInputError("Failed to parse audio", str(e))

    def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw WAV audio bytes

        Returns:
            Dict with transcript, language, confidence, duration
        """
        start_time = time.perf_counter()
        self._lazy_load()

        audio_array, sample_rate = self.validate_audio(audio_bytes)
        duration = len(audio_array) / sample_rate

        try:
            # Write to temp file for pipeline compatibility
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    pcm = (audio_array * 32767).astype(np.int16)
                    wf.writeframes(pcm.tobytes())

            result = self._pipeline(
                tmp_path,
                generate_kwargs={
                    "language": self.config.get("language", "en"),
                    "task": self.config.get("task", "transcribe"),
                },
            )

            Path(tmp_path).unlink(missing_ok=True)

            transcript = result.get("text", "").strip()
            if not transcript:
                raise ASRError("Empty transcription result — audio may be silent or too noisy")

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"ASR completed in {elapsed:.1f}ms: '{transcript[:60]}...'")

            return {
                "transcript": transcript,
                "language": self.config.get("language", "en"),
                "confidence": None,  # Whisper doesn't expose per-utterance confidence
                "duration_seconds": round(duration, 2),
                "processing_time_ms": round(elapsed, 2),
            }

        except ASRError:
            raise
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}", exc_info=True)
            raise ASRError("Transcription failed", str(e))

    def compute_wer(self, references: list, hypotheses: list) -> Dict[str, float]:
        """
        Compute Word Error Rate on a test set.

        Args:
            references: List of ground-truth transcripts
            hypotheses: List of ASR output transcripts

        Returns:
            Dict with WER and breakdown
        """
        try:
            from jiwer import wer, cer
        except ImportError:
            logger.warning("jiwer not installed, using simple WER calculation")
            return self._simple_wer(references, hypotheses)

        word_error_rate = wer(references, hypotheses)
        char_error_rate = cer(references, hypotheses)

        return {
            "wer": round(word_error_rate, 4),
            "cer": round(char_error_rate, 4),
            "num_samples": len(references),
        }

    def _simple_wer(self, references: list, hypotheses: list) -> Dict[str, float]:
        """Simple WER calculation without jiwer."""
        total_words = 0
        total_errors = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            total_words += len(ref_words)
            # Simple edit distance approximation
            errors = abs(len(ref_words) - len(hyp_words))
            for r, h in zip(ref_words, hyp_words):
                if r != h:
                    errors += 1
            total_errors += errors
        wer = total_errors / max(total_words, 1)
        return {"wer": round(wer, 4), "cer": None, "num_samples": len(references)}
