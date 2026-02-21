from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router as api_router
from app.models.detector import VoiceDetector
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Request, HTTPException
from fastapi.staticfiles import StaticFiles
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing application...")
    VoiceDetector.get_instance()
    yield
    print("Shutting down...")

app = FastAPI(
    title="AI Voice Detector API",
    description="Hackathon API for detecting AI-generated speech using Wav2Vec2",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid API key or malformed request"},
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

app.include_router(api_router, prefix="")

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Voice Detector API", "endpoints": ["/health", "/detect", "/api/voice-detection"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
