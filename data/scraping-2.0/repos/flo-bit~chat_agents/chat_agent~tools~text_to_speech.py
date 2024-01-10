import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)


async def text_to_speech(agent, text: str, path: str, model="tts-1", voice="alloy"):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )

    response.stream_to_file(path)

    return "audio saved to " + path


async def texts_to_speeches(agent, texts: list, paths: list, model="tts-1", voice="alloy"):
    output = ""
    for text, path in zip(texts, paths):
        output += await text_to_speech(text, path, model, voice) + "\n"

    return output

tool_text_to_speech = {
    "info": {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Creates an audio file from given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to audio",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to save audio to, relative to the current working directory"
                    },
                    "model": {
                        "type": "string",
                        "description": "model to use for audio generation, defaults to tts-1",
                        "enum": ["tts-1", "tts-1-hd"]
                    },
                    "voice": {
                        "type": "string",
                        "description": "voice to use for audio generation, defaults to alloy",
                        "enum": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    }
                },
                "required": ["text", "path"],
            },
        }
    },
    "function": text_to_speech,
}

tool_texts_to_speeches = {
    "info": {
        "type": "function",
        "function": {
            "name": "texts_to_speeches",
            "description": "Creates audio files from given texts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "description": "Texts to convert to audio",
                        "items": {
                            "type": "string"
                        }
                    },
                    "paths": {
                        "type": "array",
                        "description": "Paths to save audio to, relative to the current working directory",
                        "items": {
                            "type": "string"
                        }
                    },
                    "model": {
                        "type": "string",
                        "description": "model to use for audio generation, defaults to tts-1",
                        "enum": ["tts-1", "tts-1-hd"]
                    },
                    "voice": {
                        "type": "string",
                        "description": "voice to use for audio generation, defaults to alloy",
                        "enum": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                    }
                },
                "required": ["texts", "paths"],
            },
        }
    },
    "function": texts_to_speeches,
}
