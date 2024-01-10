import queue, sys
import sounddevice as sd
import numpy as np
import asyncio
import audio
import api
import openai
from pydub import AudioSegment
from io import BytesIO

# 録音の設定
SAMPLE_RATE = 16000
CHANNELS = 1
sd.default.device = 8

REC_FILE = "../sound/rec.wav"

# ChatGPTのキャラクター設定
CHAT_CHARACTER = '''あなたはなんでも解説してくれる博士です。6行で説明してください。ただし一文ごとに/を入れて話してください。文末にも/を入れてください。'''

codama = audio.Audio(SAMPLE_RATE, CHANNELS)
myopenai = api.OpenAI()
mygoogle = api.Google()
pp = api.Porcupine()

# ウェイクアップワード検知でストリーミングする際のキュー
q = queue.Queue()

# 非同期でChatGPTの返答内容をキューに追加
async def chat_worker(text_queue):
    async for chunk in await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": CHAT_CHARACTER},
            {"role": "user", "content": input_text}
        ],
        temperature=0.8,
        stream=True
    ):
        content = chunk['choices'][0]['delta'].get('content')
        if content:
            print(content)
            await text_queue.put(content)

# 返答内容をキューから非同期で取得して音声合成
async def speech_worker(text_queue):
    word = ''
    while True:
        text = await text_queue.get()
        if text is None:
            break

        # "/"ごとに区切って音声合成
        word += text
        if '/' in word:
            word = word.strip('/')
            print(word)
            response = mygoogle.synthesize_response(word)
            audio_segment = AudioSegment.from_mp3(BytesIO(response.audio_content))

            sd.play(np.array(audio_segment.get_array_of_samples(), dtype=np.int16), SAMPLE_RATE)
            sd.wait()
            word = ''

def recordCallback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

async def main():
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=pp.porcupine.frame_length,
            channels=CHANNELS,
            callback=recordCallback,
        )
        stream.start()
        print("Start")

        while True:
            if not q.empty():
                data = q.get(block=False)
                data = np.reshape(data, [data.shape[0]])
                
                keyword_index = pp.porcupine.process(data)
                
                # "こだま"を検知したら
                if keyword_index == 0:
                    print("Detected: こだま")
                    stream.stop()
                    stream.close()

                    # 3秒間録音
                    codama.record(REC_FILE, 3)
                    # Whisper APIで音声データをテキストに変換
                    global input_text
                    input_text = myopenai.whisper(REC_FILE)

                    # ChatGPTの返答内容を入れるキュー
                    text_queue = asyncio.Queue()

                    chat_task = asyncio.create_task(chat_worker(text_queue))
                    speech_task = asyncio.create_task(speech_worker(text_queue))

                    await chat_task
                    await text_queue.put(None)
                    await speech_task

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
    asyncio.run(main())