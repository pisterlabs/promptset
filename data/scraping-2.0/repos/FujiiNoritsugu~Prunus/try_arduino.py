import os
import openai

import re
import subprocess
import speech_recognition
import time
import pyaudio

SAMPLERATE = 44100

def extract_code(text):
    # プログラムコードの正規表現パターン
    code_pattern = r"```[^\n]*\n([\s\S]*?)\n```"

    # 正規表現パターンにマッチする全ての部分を抽出
    code_matches = re.findall(code_pattern, text)

    # 抽出されたコードをリストで返す
    return code_matches


def remove_2_byte_characters(input_text):
    # 正規表現パターンでASCII範囲外の文字を検出
    pattern = re.compile("[^\x00-\x7F]+")

    # パターンに一致する文字列を空白文字列に置換
    cleaned_text = pattern.sub("", input_text)

    return cleaned_text

def callback(in_data, frame_count, time_info, status):
    global sprec

    try:
        audiodata = speech_recognition.AudioData(in_data,SAMPLERATE,2)
        sprec_text = sprec.recognize_google(audiodata, language='ja-JP')
        print(sprec_text)

        operation = f"Arduino UNO で{sprec_text}プログラムを生成してください。"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "あなたはArduinoのプログラムを生成するAIです。",
                },
                {
                    "role": "user",
                    "content": operation,
                },
            ],
        )

        response = completion.choices[0].message.content
        print(response)
        # コードをスケッチに書き込む
        with open('./test_sketch/test_sketch.ino', 'w') as f:
            f.writelines(extract_code(response))

        # アップロードシェルを実行する
        subprocess.run('./upload.sh', shell=True)

    except speech_recognition.UnknownValueError:
        pass
    except speech_recognition.RequestError as e:
        pass
    finally:
        return (None, pyaudio.paContinue)

def main():
    # ファイルを開く
    # with open(os.getenv("OPENAI_API_KEY"), "r") as file:
    with open("../chat_gpt_api_key", "r") as file:
        # ファイルからデータを読み込む
        data = file.read()

    # 読み込んだデータを変数に設定
    openai.api_key = data

    global sprec 
    sprec = speech_recognition.Recognizer()  # インスタンスを生成
    # Audio インスタンス取得
    audio = pyaudio.PyAudio() 
    stream = audio.open( format = pyaudio.paInt16,
                        rate = SAMPLERATE,
                        channels = 1, 
                        input_device_index = 11,
                        input = True, 
                        frames_per_buffer = SAMPLERATE*5, # 5秒周期でコールバック
                        stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    main()
