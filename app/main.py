"""
VoiceBot FastAPI application.
Exposes REST API endpoints for speech processing pipeline.
"""
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from app.config import get_config
from app.logger import setup_logging, get_logger, set_request_id
from app.schemas import (
    TranscribeResponse,
    IntentResponse,
    IntentPrediction,
    ResponseGenerateRequest,
    ResponseGenerateResponse,
    SynthesizeRequest,
    VoicebotResponse,
    HealthResponse,
    ErrorResponse,
    TextRequest,
)
from app.exceptions import VoiceBotError

# Initialize config
config = get_config()

# Setup logging
setup_logging(
    log_level=config.app.get("log_level", "INFO"),
    log_file=config.app.get("log_file", "logs/voicebot.log"),
)
logger = get_logger(__name__)

# Module singletons (lazy loaded)
_asr = None
_classifier = None
_generator = None
_tts = None
_start_time = time.time()


def get_asr():
    global _asr
    if _asr is None:
        from app.asr import ASRModule
        _asr = ASRModule()
    return _asr


def get_classifier():
    global _classifier
    if _classifier is None:
        from app.intent_classifier import IntentClassifier
        _classifier = IntentClassifier()
    return _classifier


def get_generator():
    global _generator
    if _generator is None:
        from app.response_generator import ResponseGenerator
        _generator = ResponseGenerator()
    return _generator


def get_tts():
    global _tts
    if _tts is None:
        from app.tts import TTSModule
        _tts = TTSModule()
    return _tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info(f"Starting {config.app.get('name', 'VoiceBot')} v{config.app.get('version', '1.0.0')}")
    logger.info("=" * 60)
    # Warm up modules
    try:
        get_classifier()._lazy_load()
        get_generator()._lazy_load()
        logger.info("Core modules initialized")
    except Exception as e:
        logger.warning(f"Startup warm-up warning: {e}")
    yield
    logger.info("VoiceBot shutting down")


app = FastAPI(
    title="VoiceBot Customer Support API",
    description=(
        "Production-ready Voice Bot system for customer support. "
        "Processes audio through ASR → Intent Classification → Response Generation → TTS."
    ),
    version=config.app.get("version", "1.0.0"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    request_id = set_request_id()
    request.state.request_id = request_id
    request.state.start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    elapsed = (time.perf_counter() - request.state.start_time) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response


@app.exception_handler(VoiceBotError)
async def voicebot_error_handler(request: Request, exc: VoiceBotError):
    request_id = getattr(request.state, "request_id", "N/A")
    logger.error(f"VoiceBotError: {exc.message} | {exc.details}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.message,
            details=exc.details,
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "N/A")
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            request_id=request_id,
        ).model_dump(),
    )


# ─────────────────────────── HEALTH ──────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check."""
    classifier = get_classifier()
    generator = get_generator()
    tts = get_tts()
    asr = get_asr()
    return HealthResponse(
        status="healthy",
        version=config.app.get("version", "1.0.0"),
        models_loaded={
            "asr": asr.is_loaded,
            "intent_classifier": classifier.is_loaded,
            "response_generator": generator.is_loaded,
            "tts": tts.is_loaded,
        },
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ─────────────────────────── ASR ──────────────────────────────


@app.post("/transcribe", response_model=TranscribeResponse, tags=["ASR"])
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(..., description="WAV audio file"),
):
    """
    Transcribe audio file to text using Whisper ASR.

    - Accepts WAV format
    - Returns transcript with confidence and duration
    """
    request_id = getattr(request.state, "request_id", set_request_id())

    if not audio.filename or not audio.filename.lower().endswith(".wav"):
        from app.exceptions import AudioInputError
        raise AudioInputError("Only WAV format is supported", f"Received: {audio.filename}")

    audio_bytes = await audio.read()
    logger.info(f"Received audio: {audio.filename} ({len(audio_bytes)} bytes)")

    asr = get_asr()
    result = asr.transcribe(audio_bytes)

    return TranscribeResponse(
        transcript=result["transcript"],
        language=result["language"],
        confidence=result["confidence"],
        duration_seconds=result["duration_seconds"],
        request_id=request_id,
        processing_time_ms=result["processing_time_ms"],
    )


# ─────────────────────────── INTENT ──────────────────────────────


@app.post("/predict-intent", response_model=IntentResponse, tags=["NLP"])
async def predict_intent(request: Request, body: TextRequest):
    """
    Classify user intent from text.

    Returns top intent, confidence score, and all intent probabilities.
    """
    request_id = getattr(request.state, "request_id", set_request_id())

    classifier = get_classifier()
    result = classifier.predict(body.text)

    top = result["top_intent"]
    all_intents = [
        IntentPrediction(
            intent=i["intent"],
            confidence=i["confidence"],
            display_name=i["display_name"],
        )
        for i in result["all_intents"]
    ]

    return IntentResponse(
        text=body.text,
        top_intent=IntentPrediction(
            intent=top["intent"],
            confidence=top["confidence"],
            display_name=top["display_name"],
        ),
        all_intents=all_intents,
        is_confident=result["is_confident"],
        request_id=request_id,
        processing_time_ms=result["processing_time_ms"],
    )


# ─────────────────────────── RESPONSE ──────────────────────────────


@app.post("/generate-response", response_model=ResponseGenerateResponse, tags=["NLP"])
async def generate_response(request: Request, body: ResponseGenerateRequest):
    """
    Generate a customer support response for given text/intent.

    If intent is not provided, it will be classified automatically.
    """
    request_id = getattr(request.state, "request_id", set_request_id())

    generator = get_generator()

    if body.intent:
        # Use provided intent
        classifier = get_classifier()
        intent_result = classifier.predict(body.text)
        confidence = next(
            (i["confidence"] for i in intent_result["all_intents"] if i["intent"] == body.intent),
            0.5,
        )
        result = generator.generate(
            intent=body.intent,
            confidence=confidence,
            context=body.context,
        )
    else:
        result = generator.generate_from_text(body.text, context=body.context)

    return ResponseGenerateResponse(
        response_text=result["response_text"],
        intent_used=result["intent_used"],
        follow_up=result.get("follow_up"),
        request_id=request_id,
        processing_time_ms=result["processing_time_ms"],
    )


# ─────────────────────────── TTS ──────────────────────────────


@app.post("/synthesize", tags=["TTS"])
async def synthesize_speech(body: SynthesizeRequest):
    """
    Convert text to speech audio (MP3).

    Returns audio/mpeg binary response.
    """
    tts = get_tts()
    result = tts.synthesize(
        text=body.text,
        language=body.language,
        slow=body.slow,
    )

    return Response(
        content=result["audio_bytes"],
        media_type="audio/mpeg",
        headers={
            "X-Duration-Estimate": str(result["duration_estimate_seconds"]),
            "X-Processing-Time-Ms": str(result["processing_time_ms"]),
            "Content-Disposition": 'attachment; filename="response.mp3"',
        },
    )


# ─────────────────────────── VOICEBOT (UNIFIED) ──────────────────────────────


@app.post("/voicebot", tags=["VoiceBot"])
async def voicebot_pipeline(
    request: Request,
    audio: UploadFile = File(..., description="WAV audio input"),
    return_metadata: bool = Form(default=False, description="Include JSON metadata in headers"),
):
    """
    **Unified end-to-end VoiceBot endpoint: Audio → Audio**

    Pipeline:
    1. ASR: WAV → transcript
    2. Intent Classification: transcript → intent + confidence
    3. Response Generation: intent → response text
    4. TTS: response text → MP3 audio

    Returns MP3 audio with pipeline metadata in response headers.
    """
    request_id = getattr(request.state, "request_id", set_request_id())
    pipeline_start = time.perf_counter()
    timings: dict = {}

    # ── STEP 1: ASR ──
    if not audio.filename or not audio.filename.lower().endswith(".wav"):
        from app.exceptions import AudioInputError
        raise AudioInputError("Only WAV audio is supported for /voicebot endpoint")

    audio_bytes = await audio.read()
    logger.info(f"[{request_id}] VoiceBot pipeline started: {audio.filename} ({len(audio_bytes)} bytes)")

    t0 = time.perf_counter()
    asr = get_asr()
    asr_result = asr.transcribe(audio_bytes)
    timings["asr_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    transcript = asr_result["transcript"]
    logger.info(f"[{request_id}] ASR: '{transcript[:80]}'")

    # ── STEP 2: INTENT ──
    t0 = time.perf_counter()
    classifier = get_classifier()
    intent_result = classifier.predict(transcript)
    timings["intent_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    top_intent = intent_result["top_intent"]
    logger.info(
        f"[{request_id}] Intent: {top_intent['intent']} ({top_intent['confidence']:.3f})"
    )

    # ── STEP 3: RESPONSE ──
    t0 = time.perf_counter()
    generator = get_generator()
    response_result = generator.generate(
        intent=top_intent["intent"],
        confidence=top_intent["confidence"],
    )
    timings["response_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    response_text = response_result["response_text"]
    logger.info(f"[{request_id}] Response: '{response_text[:80]}'")

    # ── STEP 4: TTS ──
    t0 = time.perf_counter()
    tts = get_tts()
    tts_result = tts.synthesize(text=response_text)
    timings["tts_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    total_latency = round((time.perf_counter() - pipeline_start) * 1000, 2)
    logger.info(
        f"[{request_id}] Pipeline complete in {total_latency}ms "
        f"(ASR: {timings['asr_ms']}ms, Intent: {timings['intent_ms']}ms, "
        f"Response: {timings['response_ms']}ms, TTS: {timings['tts_ms']}ms)"
    )

    return Response(
        content=tts_result["audio_bytes"],
        media_type="audio/mpeg",
        headers={
            "X-Request-ID": request_id,
            "X-Transcript": transcript[:200],
            "X-Intent": top_intent["intent"],
            "X-Intent-Confidence": str(round(top_intent["confidence"], 4)),
            "X-Response-Text": response_text[:200],
            "X-Total-Latency-Ms": str(total_latency),
            "X-ASR-Ms": str(timings["asr_ms"]),
            "X-Intent-Ms": str(timings["intent_ms"]),
            "X-Response-Ms": str(timings["response_ms"]),
            "X-TTS-Ms": str(timings["tts_ms"]),
            "Content-Disposition": 'attachment; filename="voicebot_response.mp3"',
        },
    )


# ─────────────────────────── EVALUATION ──────────────────────────────


@app.post("/evaluate/intent", tags=["Evaluation"])
async def evaluate_intent_classifier(samples: list = None):
    """
    Run intent classifier evaluation and return metrics.
    Loads test data from configured path or uses provided samples.
    """
    import json
    from pathlib import Path
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix
    )

    test_path = Path(__file__).resolve().parent.parent / "data/intent_dataset.json"
    with open(test_path) as f:
        data = json.load(f)

    classifier = get_classifier()
    y_true, y_pred, confidences = [], [], []

    for item in data:
        result = classifier.predict(item["text"])
        y_true.append(item["intent"])
        y_pred.append(result["top_intent"]["intent"])
        confidences.append(result["top_intent"]["confidence"])

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    from app.intent_classifier import INTENT_LABELS
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "num_samples": len(data),
        "confusion_matrix": cm.tolist(),
        "labels": INTENT_LABELS,
        "classifier_type": "keyword_fallback" if classifier.using_fallback else "transformer",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.app.get("host", "0.0.0.0"),
        port=config.app.get("port", 8000),
        reload=config.app.get("debug", False),
        log_level=config.app.get("log_level", "INFO").lower(),
    )
