import os
import openai
import librosa
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# OpenAIのAPIキーを環境変数から取得します。
openai.api_key = os.getenv('OPENAI_API_KEY')

# mp3ファイル名を環境変数から取得します。
mp3_filename = os.getenv('MP3_FILENAME')

audio_file= open(mp3_filename, "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)

print(transcript.text)

# 結果をテキストファイルに保存します。
with open('transcription.txt', 'w') as file:
    file.write(transcript.text)
    print("Transcription saved to 'transcription.txt'")

