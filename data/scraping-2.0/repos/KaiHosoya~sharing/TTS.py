from pathlib import Path
from openai import OpenAI

#OpenAIのAPIkeyを設定
client = OpenAI(api_key="")


speech_file_path = Path(__file__).parent / "保存したいファイルの場所と名前を指定"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input='''文字起こししたい文章を挿入'''
)

response.stream_to_file(speech_file_path)
