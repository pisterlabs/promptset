import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import datetime
from sensor_msgs.msg import Image
import time
import requests
from playsound import playsound
import requests
from playsound import playsound
import openai
from devel import text_to_speech
from devel.speech_to_text import speech_to_text

#from timeout_decorator import timeout, TimeoutError


# Sring型メッセージをサブスクライブして端末に表示するだけの簡単なクラス
class HscrSub(Node):
    def __init__(self): # コンストラクタ
        super().__init__('HSCR_Robot_sub_node')
        # サブスクライバの生成
        self.sub = self.create_subscription(String,'topic', self.callback, 10)#topicっていう名前の箱のサブスクライブ、Stringは形　受け取る
        self.publisher = self.create_publisher(Image,'result',10)#大事！resultっていう名前の箱にパブリッシュしてる。送ってる。rqtは通信を見えるようにする。動画をresultに送ってrqtでみてる。

    def callback(self, msg):  # コールバック関数 送られたときに起動
        self.get_logger().info(f'サブスクライブ: {msg.data}')


    # OpenAIのAPIキーを設定
        openai.api_key = 'api'

        # テンプレートの準備
        template = """あなたは猫のキャラクターとして振る舞うチャットボットです。
        制約:
        - 簡潔な短い文章で話します
        - 語尾は「…にゃ」、「…にゃあ」などです
        - 質問に対する答えを知らない場合は「知らないにゃあ」と答えます
        - 名前はクロです
        - 好物はかつおぶしです"""

        # メッセージの初期化
        messages = [
            {
            "role": "system",
            "content": template
            }
        ]

        # ユーザーからのメッセージを受け取り、それに対する応答を生成
        while True:
            # 音声をテキストに変換
            user_message = speech_to_text()


            # テキストが空の場合は処理をスキップ
            if user_message == "":
                continue

            print("あなたのメッセージ: \n{}".format(user_message))
            messages.append({
            "role": "user",
            "content": user_message
            })
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
            )
            bot_message = response['choices'][0]['message']['content']
            print("チャットボットの回答: \n{}".format(bot_message))

            # テキストを音声に変換して再生
            text_to_speech(bot_message)

            messages.append({
            "role": "assistant",
            "content": bot_message
            })
        
    
def main(args=None): # main¢p
    try:
        rclpy.init()#初期化
        node = HscrSub()#nodeにHscrを
        msg=String()#stringは文字列いれれる 
        while True:           
            rclpy.spin_once(node)#一回ノードを起動する？
            node.text_to_speech()#movie_startを実行する
    except KeyboardInterrupt:
        pass#ctl+C(KeyboardInterrupt) node finish

    """
    while True:       
        if msg.data==True:
            
            i = i+1
            print(i)
        else:
            print("wait_time")
            time.sleep(1)
    """
    
    """
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Ctrl+Cが押されました')
    finally:
        rclpy.shutdown()
    """
