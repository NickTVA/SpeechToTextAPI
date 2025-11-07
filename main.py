import io
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import numpy as np
import wave
import logging
import time
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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

@app.post("/transcribe/wav")
async def transcribe_wav_audio(file: bytes = File(...)):
    start_time = time.time()
    request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{int(time.time() * 1000) % 1000}"

    try:
        # Log incoming request
        file_size = len(file) if file else 0
        logger.info(f"[{request_id}] Received transcription request - File size: {file_size:,} bytes")

        if not file:
            logger.warning(f"[{request_id}] No audio data provided")
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
                duration = n_frames / framerate if framerate > 0 else 0

                # Log audio file details
                logger.info(f"[{request_id}] Audio details - Channels: {channels}, "
                           f"Sample width: {sample_width*8} bit, Sample rate: {framerate} Hz, "
                           f"Duration: {duration:.2f} seconds")

                if channels > 2:
                    logger.warning(f"[{request_id}] Invalid channels: {channels}")
                    raise HTTPException(status_code=400, detail="Audio must be mono or stereo")
                if sample_width not in [1, 2, 3, 4]:
                    logger.warning(f"[{request_id}] Unsupported sample width: {sample_width}")
                    raise HTTPException(status_code=400, detail="Unsupported sample width")
                if framerate < 8000 or framerate > 48000:
                    logger.warning(f"[{request_id}] Invalid sample rate: {framerate}")
                    raise HTTPException(status_code=400, detail="Sample rate must be between 8000 and 48000 Hz")
        except wave.Error as e:
            logger.error(f"[{request_id}] Invalid WAV file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid WAV file: {str(e)}")

        # Log start of transcription
        logger.info(f"[{request_id}] Starting transcription...")
        transcription_start = time.time()

        model = load_model()
        result = model.transcribe(tmp_file_path)

        transcription_time = time.time() - transcription_start

        # Log transcription results
        transcribed_text = result["text"]
        detected_language = result.get("language", "unknown")
        word_count = len(transcribed_text.split()) if transcribed_text else 0

        logger.info(f"[{request_id}] Transcription completed - "
                   f"Language: {detected_language}, "
                   f"Word count: {word_count}, "
                   f"Transcription time: {transcription_time:.2f}s")

        # Log the actual transcription (truncate if too long for logging)
        if transcribed_text:
            log_text = transcribed_text[:500] + "..." if len(transcribed_text) > 500 else transcribed_text
            logger.info(f"[{request_id}] Transcribed text: {log_text.strip()}")
        else:
            logger.warning(f"[{request_id}] Empty transcription result")

        import os
        os.unlink(tmp_file_path)

        # Log total processing time
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Request completed - Total processing time: {total_time:.2f}s")

        return JSONResponse(content={
            "text": result["text"],
            "language": result.get("language", "unknown")
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe/ogg")
async def transcribe_ogg_audio(file: bytes = File(...)):
    start_time = time.time()
    request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{int(time.time() * 1000) % 1000}"

    try:
        # Log incoming request
        file_size = len(file) if file else 0
        logger.info(f"[{request_id}] Received OGG transcription request - File size: {file_size:,} bytes")

        if not file:
            logger.warning(f"[{request_id}] No audio data provided")
            raise HTTPException(status_code=400, detail="No audio data provided")

        # Validate OGG file header
        if len(file) < 4 or file[:4] != b'OggS':
            logger.warning(f"[{request_id}] Invalid OGG file - missing OggS header")
            raise HTTPException(status_code=400, detail="Invalid OGG file format")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        try:
            # Log start of transcription
            logger.info(f"[{request_id}] Starting transcription of OGG file...")
            transcription_start = time.time()

            model = load_model()
            result = model.transcribe(tmp_file_path)

            transcription_time = time.time() - transcription_start

            # Log transcription results
            transcribed_text = result["text"]
            detected_language = result.get("language", "unknown")
            word_count = len(transcribed_text.split()) if transcribed_text else 0

            logger.info(f"[{request_id}] Transcription completed - "
                       f"Language: {detected_language}, "
                       f"Word count: {word_count}, "
                       f"Transcription time: {transcription_time:.2f}s")

            # Log the actual transcription (truncate if too long for logging)
            if transcribed_text:
                log_text = transcribed_text[:500] + "..." if len(transcribed_text) > 500 else transcribed_text
                logger.info(f"[{request_id}] Transcribed text: {log_text.strip()}")
            else:
                logger.warning(f"[{request_id}] Empty transcription result")

            import os
            os.unlink(tmp_file_path)

            # Log total processing time
            total_time = time.time() - start_time
            logger.info(f"[{request_id}] Request completed - Total processing time: {total_time:.2f}s")

            return JSONResponse(content={
                "text": result["text"],
                "language": result.get("language", "unknown")
            })

        except Exception as e:
            import os
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            raise e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}