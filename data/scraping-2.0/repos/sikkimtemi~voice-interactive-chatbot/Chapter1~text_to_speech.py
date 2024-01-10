from openai import OpenAI
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import numpy as np
import threading
import time

client = OpenAI()
play_queue = []
playing = False


def play_audio(audio_data):
    global playing

    # 音声データをキューに追加
    play_queue.append(audio_data)

    # 再生プロセスが稼働していない場合、開始する
    if not playing:
        playing = True
        threading.Thread(target=process_queue).start()


def process_queue():
    global playing

    while play_queue:
        # キューから次の音声データを取得
        audio_data = play_queue.pop(0)
        sig, sr = sf.read(audio_data, always_2d=True)
        channels = sig.shape[1]

        # オーディオストリームの設定
        with sd.OutputStream(
            samplerate=sr, blocksize=1024, channels=channels, dtype=np.float32
        ) as stream:
            current_frame = 0
            while current_frame < len(sig):
                chunksize = min(len(sig) - current_frame, 1024)
                chunk = sig[current_frame : current_frame + chunksize]
                stream.write(chunk.astype(np.float32))
                current_frame += chunksize

        # キューが空になったらスリープして再チェック
        if not play_queue:
            time.sleep(0.1)

    # 再生プロセス終了
    playing = False


def text_to_speech(text):
    # 音声合成する
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )

    # 音声データを取得
    audio_buffer = io.BytesIO(response.content)

    # 音声データを再生
    play_audio(audio_buffer)


if __name__ == "__main__":
    # 音声に変換したいテキスト
    text1 = "私は音声対話型チャットボットです。"
    text2 = "なにかお手伝いできることはありますか？"

    print("text1を音声合成中...")
    text_to_speech(text1)
    print("text2を音声合成中...")
    text_to_speech(text2)
