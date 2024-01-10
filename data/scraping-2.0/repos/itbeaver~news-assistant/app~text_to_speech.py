from pathlib import Path
from openai import OpenAI

import sys

def text_to_speech(text, speech_file_path=None):
    default_path = Path(__file__).parent / "speech.mp3"
    client = OpenAI()
    speech_file_path = speech_file_path or default_path
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python text_to_speech.py 'text to convert to speech' [output_file_path]")
        sys.exit(1)
    text = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else None

    speech_file = text_to_speech(text, file_path)
    print(f"Speech file created at: {speech_file}")

