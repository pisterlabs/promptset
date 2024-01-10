import io
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
# python -m pip install python-dotenv
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
client.api_key = os.environ.get("OPENAI_API_KEY")

def stream_and_play(text):
  response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text,
  )

  # 将二进制响应内容转换为字节流。
  byte_stream = io.BytesIO(response.content)
  # 从字节流中读取音频数据。
  audio = AudioSegment.from_file(byte_stream, format="mp3")
  # 播放
  play(audio)
if __name__ == "__main__":
  text = input("输入您要播放的文本: ")
  stream_and_play(text)