import rclpy                         # ROS2のPythonモジュール
from rclpy.node import Node          # rclpy.nodeモジュールからNodeクラスをインポート
from std_msgs.msg import String      # std_msgs.msgモジュールからStringクラスをインポート
import numpy as np
import sounddevice as sd
import threading
import time
from scipy.io.wavfile import write
import openai
from .record_test import record_main

class HscrPub(Node):  # "Happy World"とパブリッシュ並びに表示するクラス
    def __init__(self):  # コンストラクタ
        super().__init__('HSCR_Robot_pub_node')
        self.pub = self.create_publisher(String, 'topic', 10)   # パブリッシャの生成
        self.create_timer(1.0, self.callback)

    def callback(self):  # コールバック関数
        print("a")
        msg = String()
        msg.data = input()
        self.pub.publish(msg)
        self.get_logger().info(f'パブリッシュ: {msg.data}')

def main(args=None):  # main関数
    rclpy.init()
    node = HscrPub()
    # OpenAIのAPIキーを設定
    openai.api_key = 'api'
    record_main()
    print("テキストファイル作成")
    with open('/home/uchida/devel1/src/devel/devel/output.wav', "rb") as audio_file:
    # Whisper APIを使用してオーディオファイルをテキストに変換
        #transcript = openai.Audio.transcribe("whisper-1", audio_file)

        # 音声からテキスト変換した結果をファイルに保存
        try:
            with open('/home/uchida/devel1/src/devel/devel/enter_voice_word.txt', 'w') as output_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                output_file.write(transcript.text)
                print("transcript.text:", transcript.text)
                file_path = '/home/uchida/devel1/src/devel/devel/enter_voice_word.txt'
                with open(file_path, 'r') as file:
                    file_content = file.read()
                print(file_content)
        except FileNotFoundError:
            print(f"ファイル '{file_path}' が見つかりません。")
        except Exception as e:
            print(f"エラー: {e}")

    try:
        rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
