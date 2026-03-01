"""
Pydantic schemas for VoiceBot API request/response models.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    """Response from /transcribe endpoint."""
    transcript: str = Field(..., description="Recognized text from audio")
    language: str = Field(default="en", description="Detected language")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="ASR confidence score")
    duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds")
    request_id: str = Field(..., description="Request trace ID")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class IntentPrediction(BaseModel):
    """Single intent prediction with confidence."""
    intent: str = Field(..., description="Predicted intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    display_name: str = Field(..., description="Human-readable intent name")


class IntentResponse(BaseModel):
    """Response from /predict-intent endpoint."""
    text: str = Field(..., description="Input text that was classified")
    top_intent: IntentPrediction = Field(..., description="Highest confidence intent")
    all_intents: List[IntentPrediction] = Field(..., description="All intents with scores")
    is_confident: bool = Field(..., description="Whether confidence exceeds threshold")
    request_id: str = Field(..., description="Request trace ID")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ResponseGenerateRequest(BaseModel):
    """Request body for /generate-response endpoint."""
    text: str = Field(..., min_length=1, description="User input text")
    intent: Optional[str] = Field(None, description="Pre-computed intent (optional)")
    context: Optional[Dict[str, Any]] = Field(None, description="Conversation context")


class ResponseGenerateResponse(BaseModel):
    """Response from /generate-response endpoint."""
    response_text: str = Field(..., description="Generated response")
    intent_used: str = Field(..., description="Intent used to generate response")
    follow_up: Optional[str] = Field(None, description="Follow-up suggestion")
    request_id: str = Field(..., description="Request trace ID")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class SynthesizeRequest(BaseModel):
    """Request body for /synthesize endpoint."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    language: str = Field(default="en", description="Language code")
    slow: bool = Field(default=False, description="Use slower speech rate")


class VoicebotResponse(BaseModel):
    """Response metadata from /voicebot endpoint (audio returned as file)."""
    transcript: str = Field(..., description="ASR transcript")
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Intent confidence")
    response_text: str = Field(..., description="Generated text response")
    request_id: str = Field(..., description="Request trace ID")
    total_latency_ms: float = Field(..., description="End-to-end processing time")
    pipeline_breakdown: Dict[str, float] = Field(..., description="Per-stage timing")


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    details: Optional[str] = None
    request_id: Optional[str] = None


class TextRequest(BaseModel):
    """Simple text input request."""
    text: str = Field(..., min_length=1, description="Input text")
