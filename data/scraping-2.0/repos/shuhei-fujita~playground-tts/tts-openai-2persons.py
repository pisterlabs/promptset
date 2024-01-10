from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydub import AudioSegment

def load_text(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()

def generate_speech(client, text, model_name="tts-1", voice="nova"):
    response = client.audio.speech.create(model=model_name, voice=voice, input=text)
    return response

def process_speech(client, text, voice):
    response = generate_speech(client, text, voice=voice)
    temp_path = Path(__file__).parent / f"speech_{voice}.mp3"
    response.stream_to_file(temp_path)
    return AudioSegment.from_mp3(temp_path)

# 環境変数のロード]
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# テキストファイルの読み込み
text_file_path = Path(__file__).parent / "sample-text-jp.txt"
text = load_text(text_file_path)

combined_sounds = AudioSegment.empty()
voice_mapping = {
    # 必要に応じてスピーカーと音声を追加
    "Aさん": "nova",
    "Bさん": "alloy",
    "Kim Chae Won": "shimmer",
    "Park Jimin": "fable",
    "Kim Taehyung": "echo",
}

for line in text.split("\n"):
    for speaker, voice in voice_mapping.items():
        if f"{speaker}:" in line:
            print(f"処理中: {speaker} のセリフ - ボイス: {voice}")
            combined_sounds += process_speech(
                client, line.replace(f"{speaker}:", "").strip(), voice
            )

# 結合した音声をMP3ファイルとして保存
speech_file_path = Path(__file__).parent / "speech_combined.mp3"
combined_sounds.export(speech_file_path, format="mp3")
