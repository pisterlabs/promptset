"""Create speech from text"""
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def text_to_speech(text: str, path: str) -> str:
    """
    Creates speech from text.
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    response.stream_to_file(path)
    return f"Speech saved to {path} successfully."


available_functions = {
    "text_to_speech": text_to_speech
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Creates speech from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to convert to speech."
                    },
                    "path": {
                        "type": "string",
                        "description": "The path to save the speech to."
                    }
                },
                "required": ["text", "path"]
            }
        }
    }
]
