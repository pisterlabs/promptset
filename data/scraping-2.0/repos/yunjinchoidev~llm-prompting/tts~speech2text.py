# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
from load_dotenv import load_dotenv

load_dotenv()
import os


def transcribe_audio(audio_file):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    text_eng = transcript["text"]
    return text_eng


if __name__ == "__main__":
    audio_file = open("./whisper/audio.mp3", "rb")
    text_eng = transcribe_audio(audio_file)
    print(text_eng)

    # 저장
    with open("./whisper/transcript.txt", "w") as f:
        f.write(text_eng)
