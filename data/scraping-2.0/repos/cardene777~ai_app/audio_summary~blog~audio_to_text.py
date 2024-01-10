import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_audio_text(file_path: str) -> str:
    audio_file = open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def main():
    file_path = "./audio.mp3"
    audio_text = get_audio_text(file_path)
    print(audio_text)


if __name__ == "__main__":
    main()
