import os
import openai
from typing import Optional

def set_openai_api_key(api_key: Optional[str] = None) -> None:
    """Set OpenAI API key
    
    Args:
        api_key (Optional[str]): OpenAI API key. API key will be retrieved 
            from environment variable OPENAI_API_KEY if not provided.
        
    """

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = api_key
    
class WhisperExtention:
    def __init__(self, api_key: Optional[str] = None) -> None:
        if openai.api_key is None:
            set_openai_api_key(api_key)
    
    def transcribe_file(self, file):
        with open(file, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript['text']
        