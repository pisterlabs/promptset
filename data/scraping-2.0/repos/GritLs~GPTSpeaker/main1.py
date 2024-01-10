from speeach_module.wakeup import VoiceWakeUp
from speeach_module.speech2text import BaiduASR
from speeach_module.text2speech import BaiduTTS
from gpt_module.chat import LangChainAgent
import os
import threading
import time
from iodevices.LED import LED
os.environ["SERPER_API_KEY"] = "" # 你的serper key
keyword_path = './speechmodules/Hey-Murphy_en_mac_v2_1_0.ppn'  # 你的唤醒词检测离线文件地址
wake_model = '/home/pi/Desktop/GPTSpeaker/speeach_module/MySnowboy/resources/models/HeyMurphy.pmdl' # 中文模型地址
Baidu_APP_ID = '32714532'  # 你的百度APP_ID
Baidu_API_KEY = 'GgWZBkVHMtZb1dmpH3POKGB7'  # 你的百度API_KEY
Baidu_SECRET_KEY = 'T2ewdGvihXBKEykoNuhpGhdufz3EOIqQ'  # 你的百度



import threading
import time

# 新建一个函数用来控制 LED 灯的闪烁
def led_blink(led, stop_event):
    while not stop_event.is_set():
        led.on()
        time.sleep(0.5)
        led.off()
        time.sleep(0.5)

def run(voiceWakeUp, asr, tts):
      keyword_idx = voiceWakeUp.start()
      if keyword_idx:
        voiceWakeUp.terminate() 
        openai_chat_module = LangChainAgent()
        tts.text_to_speech_and_play("嗯,我在,请讲！")
        while True:
            q = asr.speech_to_text()
            print(f'recognize_from_microphone, text={q}')

            # 创建一个事件，用来告知 LED 灯何时停止闪烁
            stop_blinking = threading.Event()
            # 创建一个 LED 对象，传入的参数是 LED 接入的 GPIO 引脚编号
            led = LED(11)
            # 开始一个新的线程，让 LED 灯开始闪烁
            led_thread = threading.Thread(target=led_blink, args=(led, stop_blinking))
            led_thread.start()

            res = openai_chat_module.response(q)
            
            # openai_chat_module.response(q) 结束后，设置事件，告知 LED 灯停止闪烁
            stop_blinking.set()
            # 等待 LED 灯的线程结束
            led_thread.join()
            # 清理 LED 对象
            led.cleanup()

            print(res)
            tts.text_to_speech_and_play('嗯' + res)


def Orator():
    while True:
        sensitivity = 0.9
        audio_gain = 1
        voiceWakeUp = VoiceWakeUp(wake_model, sensitivity=sensitivity, audio_gain=audio_gain)
        asr = BaiduASR(Baidu_APP_ID, Baidu_API_KEY, Baidu_SECRET_KEY)
        tts = BaiduTTS(Baidu_APP_ID, Baidu_API_KEY, Baidu_SECRET_KEY)
        # LED灯
        try:
            run(voiceWakeUp, asr, tts)
        except KeyboardInterrupt:
            print("中断检测")
            exit(0)
        finally:
            print('本轮对话结束')
            tts.text_to_speech_and_play('嗯' + '主人，我退下啦！')
            print("中断检测")
            voiceWakeUp.terminate() 

if __name__ == '__main__':
    while True:
        Orator()