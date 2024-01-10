# API経由のwhisperのテスト
import os
import openai

try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
except:
    from dotenv import load_dotenv
    dotenv_path = '../.env'
    load_dotenv(dotenv_path)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


openai.api_key = OPENAI_API_KEY

def recognize_audio(audio):
    # audioが安静データそのものの場合の処理を書く
    with open(audio, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f, language="ja")
    return transcript["text"]
