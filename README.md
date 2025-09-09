# Speech-to-Text API

A FastAPI-based REST API for converting speech to text using OpenAI's Whisper model.

## Features

- Fast transcription using Whisper Turbo model
- Accepts WAV format audio files
- RESTful API design
- Automatic model caching for improved performance
- Comprehensive error handling

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SpeechToTextAPI
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

Start the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

The API will be available at `http://localhost:8001`

## API Documentation

### Endpoints

#### POST /transcribe
Transcribe audio from a WAV file to text.

**Request:**
- Method: `POST`
- Content-Type: `application/octet-stream`
- Body: Binary WAV audio data

**Response:**
```json
{
  "text": "Transcribed text content",
  "language": "en"
}
```

**Status Codes:**
- 200: Success
- 400: Invalid audio format or parameters
- 500: Server error

**Audio Requirements:**
- Format: WAV
- Channels: Mono or Stereo
- Sample Rate: 8001-48001 Hz
- Sample Width: 8, 16, 24, or 32 bit

#### GET /
Root endpoint returning API information.

**Response:**
```json
{
  "message": "Speech-to-Text API",
  "version": "1.0.0",
  "model": "whisper-turbo"
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Example Usage

### Using curl
```bash
curl -X POST http://localhost:8001/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

### Using Python
```python
import requests

with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/transcribe',
        data=f.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    
print(response.json())
```

## Interactive API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Model Information

This API uses OpenAI's Whisper Turbo model by default, which provides:
- Fast transcription speed
- Good accuracy for multiple languages
- Automatic language detection

## Performance Considerations

- The Whisper model is loaded once at startup and cached in memory
- First request may take longer due to model initialization
- Subsequent requests will be faster

## Error Handling

The API provides detailed error messages for common issues:
- Invalid WAV format
- Unsupported audio parameters
- File size limitations
- Transcription failures

## Requirements

- Python 3.8 - 3.11
- CUDA-capable GPU (recommended for better performance)  Need 6GB VRAM for default "medium" model
- 4GB+ RAM minimum