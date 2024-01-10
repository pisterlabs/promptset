import pvporcupine
import queue, sys
import sounddevice as sd
import numpy as np
import audio
import api
import openai
from pydub import AudioSegment
from io import BytesIO
from google.cloud import texttospeech
import os
from dotenv import load_dotenv

load_dotenv('../.env') 

AUDIO_DEVICE_NUM = 8

# 録音の設定
SAMPLE_RATE = 16000
CHANNELS = 1

REC_FILE = "../sound/rec.wav"
OUTPUT_FILE = "../sound/output.wav"

# ChatGPTのキャラクター設定
CHAT_CHARACTER = '''あなたはなんでも解説してくれる博士です。6行で説明してください。ただし一文ごとに/を入れて話してください。'''

sd.default.device = AUDIO_DEVICE_NUM

codama = audio.Audio(SAMPLE_RATE, CHANNELS)
myopenai = api.OpenAI()

openai.api_key = os.environ.get("OPEN_AI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret-key.json"
client = texttospeech.TextToSpeechClient()

# porcupineの設定
porcupine = pvporcupine.create(
  access_key=os.environ.get("ACCESS_KEY"),
  keyword_paths=["../codama_ja_raspberry-pi_v2_2_0.ppn"],
  model_path="../porcupine_params_ja.pv"
)

q = queue.Queue()

def recordCallback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def run():
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=porcupine.frame_length,
            channels=CHANNELS,
            callback=recordCallback,
        )
        stream.start()
        print("Start")

        while True:
            if not q.empty():
                data = q.get(block=False)
                data = np.reshape(data, [data.shape[0]])
                
                keyword_index = porcupine.process(data)
                
                # "こだま"を検知したら
                if keyword_index == 0:
                    print("Detected: こだま")
                    stream.stop()
                    stream.close()

                    # 3秒間録音
                    codama.record(REC_FILE, 3)
                    # Whisper APIで音声データをテキストに変換
                    input_text = myopenai.whisper(REC_FILE)

                    stream = sd.OutputStream(
                        samplerate=SAMPLE_RATE,
                        dtype="int16",
                        channels=CHANNELS,
                    )
                    stream.start()

                    # ChatGPTの返答（ストリーミングモード）
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", 
                        messages=[
                            {"role": "system", "content": CHAT_CHARACTER},
                            {"role": "user", "content": input_text}
                        ],
                        temperature=0.8,
                        stream=True
                    )

                    word = ''
                    for chunk in response:
                        content = chunk['choices'][0]['delta'].get('content')
                        if content:
                            word += content
                            if '/' in content:
                                word = word.strip('/')
                                print(word)
                                synthesis_input = texttospeech.SynthesisInput(text=word)
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="ja-JP",
                                    name="ja-JP-Wavenet-D", # 希望する音声タイプ
                                )
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3 # 希望するファイルフォーマット
                                )

                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )

                                # 音声データをPyDubのAudioSegmentに変換
                                audio_segment = AudioSegment.from_mp3(BytesIO(response.audio_content))

                                stream.write(np.array(audio_segment.get_array_of_samples(), dtype=np.int16))
                                word = ''

                    break

    except KeyboardInterrupt:
        pass
    finally:
        sd.stop()
        while not q.empty():
            q.get(block=False)
        stream.stop()
        stream.close()

if __name__ == "__main__":
    run()