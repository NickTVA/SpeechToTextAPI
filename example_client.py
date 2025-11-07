import requests
import sys
import json
from pathlib import Path

def transcribe_audio(audio_file_path, api_url="http://localhost:8001"):
    if not Path(audio_file_path).exists():
        print(f"Error: File {audio_file_path} not found")
        return None
    
    if not audio_file_path.lower().endswith('.wav'):
        print("Warning: File should be in WAV format")
    
    try:
        with open(audio_file_path, 'rb') as audio_file:
            print(f"Sending {audio_file_path} to API...")
            
            response = requests.post(
                f"{api_url}/transcribe/wav",
                data=audio_file.read(),
                headers={'Content-Type': 'application/octet-stream'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\nTranscription successful!")
                print(f"Language detected: {result.get('language', 'unknown')}")
                print(f"\nTranscribed text:\n{result['text']}")
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.json())
                return None
                
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to API at {api_url}")
        print("Make sure the API server is running")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def check_api_health(api_url="http://localhost:8001"):
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"API Status: {health['status']}")
            print(f"Model Loaded: {health['model_loaded']}")
            return True
        return False
    except:
        print("API is not reachable")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_client.py <audio_file.wav> [api_url]")
        print("\nExample:")
        print("  python example_client.py audio.wav")
        print("  python example_client.py audio.wav http://localhost:8001")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8001"
    
    print(f"Checking API health at {api_url}...")
    if check_api_health(api_url):
        print("\n" + "="*50 + "\n")
        transcribe_audio(audio_file, api_url)