from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from fastapi.responses import JSONResponse
from app.api.auth import verify_api_key
from app.utils.audio import process_audio_input
from app.models.detector import get_detector

router = APIRouter()


class DetectRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: Optional[str] = None
    message: Optional[str] = None


class DetectResponse(BaseModel):
    classification: str
    confidence_score: float
    ai_probability: float
    detected_language: str
    transcription: str
    english_translation: str
    fraud_keywords: List[str]
    overall_risk: str
    explanation: str
    audio_duration_seconds: float
    num_chunks_processed: int
    chunk_ai_probabilities: List[float]
    pitch_human_score: Optional[float] = 0.0
    pitch_std: Optional[float] = 0.0
    pitch_jitter: Optional[float] = 0.0
    smoothness_score: Optional[float] = 0.0
    variance_score: Optional[float] = 0.0
    snr_score: Optional[float] = 0.0
    heuristic_score: Optional[float] = 0.0
    debug_probs: Optional[List[float]] = []
    debug_labels: Optional[dict] = {}


@router.post("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice(request: DetectRequest):
    if not request.audio_base64 and not request.audio_url:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'audio_base64' or 'audio_url'"
        )

    try:
        audio_array, metadata = process_audio_input(
            request.audio_base64, request.audio_url, max_duration=6.0
        )
        if audio_array is None or (hasattr(audio_array, "size") and audio_array.size == 0):
            raise HTTPException(status_code=400, detail="Audio decode produced no samples")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

    try:
        detector = get_detector()
        result = detector.detect_fraud(audio_array, metadata)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/detect", response_model=DetectResponse, dependencies=[Depends(verify_api_key)])
async def detect_get(audio_url: str):
    """GET handler for audio URL-based detection."""
    request = DetectRequest(audio_url=audio_url)
    return await detect_voice(request)


# --- Hackathon Specification Endpoint ---

class HackathonRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


class HackathonResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str


async def _run_hackathon_detection(language: str, audio_format: str, audio_base64: str):
    """Shared detection logic for both POST and GET hackathon endpoints."""
    if audio_format.lower() != "mp3":
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Only mp3 format supported"}
        )

    try:
        audio_array, metadata = process_audio_input(audio_base64, None, max_duration=2.0)
        if audio_array is None or (hasattr(audio_array, "size") and audio_array.size == 0):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Audio decode produced no samples"},
            )

        detector = get_detector()
        result = detector.detect_fraud(audio_array, metadata)

        mapping = {"AI": "AI_GENERATED", "Human": "HUMAN"}
        final_class = mapping.get(result.get("classification"), "HUMAN")

        return {
            "status": "success",
            "language": language,
            "classification": final_class,
            "confidenceScore": result.get("confidence_score", 0.0),
            "explanation": result.get("explanation", "Analysis completed")
        }

    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "error", "message": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Server error: {str(e)}"}
        )


@router.post("/api/voice-detection", response_model=HackathonResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice_strict(request: HackathonRequest):
    """POST endpoint for hackathon evaluation."""
    return await _run_hackathon_detection(request.language, request.audioFormat, request.audioBase64)


@router.get("/api/voice-detection", response_model=HackathonResponse, dependencies=[Depends(verify_api_key)])
async def detect_voice_strict_get(
    language: str = Query(...),
    audioFormat: str = Query(...),
    audioBase64: str = Query(...)
):
    """GET endpoint for hackathon evaluation."""
    return await _run_hackathon_detection(language, audioFormat, audioBase64)
