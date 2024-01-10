import cv2
import base64
import os
import requests
import time
from openai import OpenAI
from collections import deque
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import threading

def play_audio_async(file_path):
    sound = AudioSegment.from_mp3(file_path)
    play(sound)

def text_to_speech(text, client):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file("output.mp3")
    threading.Thread(target=play_audio_async, args=("output.mp3",)).start()

# def text_to_speech(text, client):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text
#     )

#     # 音声データをファイルに保存
#     response.stream_to_file("output.mp3")

#     # MP3ファイルを読み込む
#     sound = AudioSegment.from_mp3("output.mp3")
#     # 音声を再生
#     play(sound)


def encode_image_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode('utf-8')

def wrap_text(text, line_length):
    """テキストを指定された長さで改行する"""
    words = text.split(' ')
    lines = []
    current_line = ''

    for word in words:
        if len(current_line) + len(word) + 1 > line_length:
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word

    lines.append(current_line)  # 最後の行を追加
    return lines

def add_text_to_frame(frame, text):
    # テキストを70文字ごとに改行
    wrapped_text = wrap_text(text, 70)

    # フレームの高さと幅を取得
    height, width = frame.shape[:2]

    # テキストのフォントとサイズ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # フォントサイズを大きくする
    color = (255, 255, 255)  # 白色
    outline_color = (0, 0, 0)  # 輪郭の色（黒）
    thickness = 2
    outline_thickness = 4  # 輪郭の太さ
    line_type = cv2.LINE_AA

    # 各行のテキストを画像に追加
    for i, line in enumerate(wrapped_text):
        position = (10, 30 + i * 30)  # 各行の位置を調整（より大きい間隔）

        # テキストの輪郭を描画
        cv2.putText(frame, line, position, font, font_scale, outline_color, outline_thickness, line_type)

        # テキストを描画
        cv2.putText(frame, line, position, font, font_scale, color, thickness, line_type)

def save_frame(frame, filename, directory='./frames'):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)

def send_frame_to_gpt(frame, previous_texts, timestamp, client):
    # 前5フレームのテキストとタイムスタンプを結合してコンテキストを作成
    context = ' '.join(previous_texts)
  
    # フレームをGPTに送信するためのメッセージペイロードを準備
    # コンテキストから前回の予測が現在の状況と一致しているかを評価し、
    # 次の予測をするように指示
    prompt_message = f"Context: {context}. Now:{timestamp}, Assess if the previous prediction matches the current situation. Current: explain the current  situation in 10 words or less. Next: Predict the next  situation in 10 words or less. Only output Current and Next"

    PROMPT_MESSAGES = {
        "role": "user",
        "content": [
            prompt_message,
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
        ],
    }

    # API呼び出しパラメータ
    params = {
        "model": "gpt-4-vision-preview",
        "messages": [PROMPT_MESSAGES],
        "max_tokens": 300,
    }

    # API呼び出し
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def send_frames_to_gpt(frames, previous_texts, timestamp, client):
    # 前5フレームのテキストとタイムスタンプを結合してコンテキストを作成
    context = ' '.join(previous_texts)
    # フレームをGPTに送信するためのメッセージペイロードを準備
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                f"Context: {context}. Now:{timestamp}, Assess if the previous prediction matches the current situation. Current: explain the current  situation in 20 words or less. Next: Predict the next  situation from current situation, context and frames in 20 words or less. Only output Current and Next",
                *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}, frames),
            ],
        },
    ]

    # API呼び出しパラメータ
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 300,
    }

    # API呼び出し
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def main():
    """メイン関数 - カメラからの映像を処理する"""
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("カメラを開くことができませんでした。")
    except IOError as e:
        print(f"エラーが発生しました: {e}")
        return

    # 最近の10フレームを保持するためのキュー
    previous_texts = deque(maxlen=10)

    base64_frames = deque(maxlen=5)


    # プログラム開始時の時間を記録
    start_time = time.time()

    while True:
        # 経過時間をチェック
        if time.time() - start_time > 300:  # 30秒経過した場合
            break

        success, frame = video.read()
        if not success:
            print("フレームの読み込みに失敗しました。")
            break

        # 現在のタイムスタンプを取得
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # フレームにタイムスタンプを追加
        timestamped_frame = frame.copy()
        cv2.putText(timestamped_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # フレームをBase64でエンコードし、キューに追加
        base64_frame = encode_image_to_base64(timestamped_frame)
        base64_frames.append(base64_frame)

        # GPTに最新の5フレームを送信し、生成されたテキストを取得
        # if len(base64_frames) == 5:
        print(len(base64_frames))
        generated_text = send_frames_to_gpt(list(base64_frames), previous_texts, timestamp, client)
        print(f"Generated Text: {generated_text}")

        # フレームにテキストを追加
        text_to_add = f"{timestamp}: {generated_text}"
        add_text_to_frame(frame, text_to_add)

        # フレームを保存
        filename = f"{timestamp}.jpg"
        save_frame(frame, filename)

        text_to_speech(generated_text, client)

        # 1秒待機
        time.sleep(1)

    # ビデオをリリースする
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()