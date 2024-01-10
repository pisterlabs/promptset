import rclpy                         # ROS2のPythonモジュール
from rclpy.node import Node          # rclpy.nodeモジュールからNodeクラスをインポート
from std_msgs.msg import String      # std_msgs.msgモジュールからStringクラスをインポート
import numpy as np
import sounddevice as sd
import threading
import time
from scipy.io.wavfile import write
import openai

class HscrPub(Node):  # "Happy World"とパブリッシュ並びに表示するクラス
    def __init__(self):  # コンストラクタ
        super().__init__('HSCR_Robot_pub_node')
        self.pub = self.create_publisher(String, 'topic', 10)   # パブリッシャの生成
        self.create_timer(1.0, self.callback)

    # OpenAIのAPIキーを設定
    openai.api_key = 'api'

    # 録音のパラメータ
    fs = 44100  # サンプルレート
    recording = np.array([])  # 録音データを保存する配列

    # 録音の開始と終了を制御するフラグ
    is_recording = False


    def record():
        """録音を行う関数"""
        global is_recording
        global recording
        while True:
            if is_recording:
                # 録音中の場合、0.5秒分の録音データを追加
                recording_chunk = sd.rec(int(0.5 * fs), samplerate=fs, channels=1)
                sd.wait()
                recording = np.append(recording, recording_chunk)
            else:
                # CPU負荷を下げるために1ミリ秒待機
                time.sleep(0.001)



        # 録音スレッドの開始
#        recording_thread = threading.Thread(target=record)
#        recording_thread.start()

#    def speech_to_text():
        """音声認識を行う関数"""
    def callback(self):  # コールバック関数
        global is_recording
        global recording
        input("Enterキーを押すと録音を開始します。\n")
        # 録音を開始
        is_recording = True
        print("録音を開始します。\n")
        input("録音中です。Enterを押すと録音を終了します。\n")
        # 録音を終了
        is_recording = False
        print("録音が終了しました。")
        if recording.size > 0:
            # 録音データが存在する場合、データをファイルに保存
            write('output.wav', fs, recording)

            # ファイルをバイナリモードで開く
            with open('output.wav', "rb") as audio_file:
                # Whisper APIを使用してオーディオファイルをテキストに変換
                transcript = openai.Audio.transcribe("whisper-1", audio_file)

            # 録音データをリセット
            recording = np.array([])

            # 音声からテキスト変換した結果を返す
            return transcript.text

        msg = String()
        msg.data = input()
        self.pub.publish(msg)
        self.get_logger().info(f'パブリッシュ: {msg.data}')
        recording_thread = threading.Thread(target=record)
        recording_thread.start()

def main(args=None):  # main関数
    rclpy.init()
    node = HscrPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
