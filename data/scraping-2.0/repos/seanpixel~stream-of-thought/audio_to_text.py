import whisper
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Whisper model   
model = whisper.load_model("base")

# Print transcription
def get_transcription(file_name):
    
    # Transcribe audio
    result = model.transcribe(file_name)

    return result['text']