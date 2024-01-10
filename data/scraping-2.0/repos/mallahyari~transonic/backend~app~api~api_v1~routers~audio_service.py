from fastapi import APIRouter, UploadFile
from app.config import settings
from openai import OpenAI
import tempfile
import structlog
import os


logger = structlog.get_logger()

audio_router = r = APIRouter()


@r.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    """Transcribe the audio file using OpenAI Whisper models

    Args:
        file (UploadFile): Audio file to be transcribed.
    """
    try:
        # Create a directory to store audio files if not exists
        os.makedirs("audio_files", exist_ok=True)

        # Save the audio file to a unique filename
        audio_path = f"audio_files/{'test'}.wav"
        # logger.info(f"{audio_path}")

        content = await file.read()
        # Create a temporary file to store the PDF content
        with tempfile.NamedTemporaryFile(dir="audio_files", suffix=".wav", delete=True) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
            
            logger.info(temp_file_path)
            client = OpenAI(api_key=settings.openai_api_key)

            audio_file= open(temp_file_path, "rb")
            transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
)
            # logger.info(f"transcript: {transcript}")
            return {"transcript": transcript.text,"message": "Audio file received and saved successfully."}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}