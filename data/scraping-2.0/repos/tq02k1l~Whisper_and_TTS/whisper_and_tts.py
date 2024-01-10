import io
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder,speech_to_text
import json

# 加载.env文件中的环境变量
load_dotenv()
client = OpenAI(
    api_key = st.secrets["OPENAI"]["OPENAI_API_KEY"],
)
def make_a_speech(input_text):
  speech_file_path = Path(__file__).parent / "speech.mp3"

  response = client.audio.speech.create(
    model="tts-1",
    voice="echo",
    input=input_text,
  )
  response.stream_to_file(speech_file_path)
  return None

st.title("Whisper and TTS")

#写一个开始录音的按钮

st.write(" :studio_microphone: Record your voice, and play the recorded audio:")
audio=mic_recorder(start_prompt="⏺️:start",stop_prompt="⏹️:stop",key='recorder')

if audio:       
    #st.audio(audio['bytes'])
    audio_bytes = audio['bytes']
    # 创建一个 AudioSegment 对象
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav") 
    # 或者转换为 WAV 格式并保存到文件
    audio.export("output.mp3", format="mp3")
    
    audio_file = open("output.mp3", "rb")
    
    transcript = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file,
      response_format="text"
    )
else:
    transcript = "Hello, I am a Translator assistant." 
     
st.write(transcript)

tag_language = st.text_input(":triangular_flag_on_post: Translation target country")
if tag_language == "":
  tag_language = "Chinese"
response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": f"You are a helpful Translator assistant, translate Chinese into {tag_language} language in a sentence and output JSON as 'translate'."},
    {"role": "user", "content": f"{transcript}"}
  ]
)
# 获取第一个 Choice 对象
first_choice = response.choices[0]
# 从 Choice 对象的 message 属性的 content 字段中解析 JSON 数据
content_json = json.loads(first_choice.message.content)
# 获取 'translate' 字段的值
translate = content_json['translate']
st.write(translate)




if translate != "":
  make_a_speech(translate)
  # 加载你的音频文件
  audio = AudioSegment.from_mp3("speech.mp3")
  # 创建一秒的静音
  silence = AudioSegment.silent(duration=2000)  # duration is in milliseconds
  # 将静音添加到音频的开始处
  delayed_audio = silence + audio
  # 将结果保存为新的音频文件
  delayed_audio.export("delayed_audio.mp3", format="mp3")


  audio_file = open('delayed_audio.mp3', 'rb')
  audio_bytes = audio_file.read()

  st.audio(audio_bytes, format='audio/mp3',start_time=0)
