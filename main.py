import io
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import numpy as np
import wave
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Speech-to-Text API",
    description="API for transcribing audio files using OpenAI Whisper",
    version="1.0.0"
)

model = None

def load_model():
    global model
    if model is None:
        logger.info("Loading Whisper medium model...")
        model = whisper.load_model("medium")
        logger.info("Model loaded successfully")
    return model

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {
        "message": "Speech-to-Text API", 
        "version": "1.0.0",
        "model": "whisper-medium"
    }

@app.post("/transcribe")
async def transcribe_audio(file: bytes = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name
        
        try:
            with wave.open(tmp_file_path, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                if channels > 2:
                    raise HTTPException(status_code=400, detail="Audio must be mono or stereo")
                if sample_width not in [1, 2, 3, 4]:
                    raise HTTPException(status_code=400, detail="Unsupported sample width")
                if framerate < 8000 or framerate > 48000:
                    raise HTTPException(status_code=400, detail="Sample rate must be between 8000 and 48000 Hz")
        except wave.Error as e:
            raise HTTPException(status_code=400, detail=f"Invalid WAV file: {str(e)}")
        
        model = load_model()
        result = model.transcribe(tmp_file_path)
        
        import os
        os.unlink(tmp_file_path)
        
        return JSONResponse(content={
            "text": result["text"],
            "language": result.get("language", "unknown")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}