from Api.openai_api import OpenAIChatbot
from Api.baidu_api_sound import BaiduSpeechRecognizer
from Api.baidu_api_text import BaiduTranslator
from Api.vits_api import voice_vits
import pygame
import io
import numpy as np
import time

class IntegratedChatbot:
    def __init__(self, openai_api_key, baidu_speech_appid, baidu_speech_api_key, baidu_speech_secret_key, baidu_translate_appid, baidu_translate_secret_key):
        self.chatbot = OpenAIChatbot(openai_api_key)
        self.recognizer = BaiduSpeechRecognizer(baidu_speech_appid, baidu_speech_api_key, baidu_speech_secret_key)
        self.translator = BaiduTranslator(baidu_translate_appid, baidu_translate_secret_key)

    def play_audio(self, file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def recognize_and_chat(self):
        while True:
            # 录制音频
            audio_data = BaiduSpeechRecognizer.record_audio(duration=5)
            
            # 将NumPy数组转换为字节流
            audio_bytes = io.BytesIO()
            np.save(audio_bytes, audio_data, allow_pickle=False)
            audio_bytes = audio_bytes.getvalue()

            # 获取token并进行语音识别
            token = self.recognizer.get_access_token()
            response = self.recognizer.recognize_speech(audio_bytes, token)

            if response.get('result'):
                recognized_text = response['result'][0]
                print("语音识别结果: ", recognized_text)

                if recognized_text.lower() == 'quit':
                    break

                # 从 OpenAI 获取回答
                openai_response = self.chatbot.get_chat_response(recognized_text)
                print("ATRI: ", openai_response)

                # 将 OpenAI 的回答翻译成日语
                translated_response = self.translator.translate(openai_response, 'zh', 'jp')
                print("翻译结果: ", translated_response)

                # 使用VITS生成语音并播放
                audio_file_path = voice_vits(translated_response)
                if audio_file_path:
                    self.play_audio(audio_file_path)

                # 等待音频播放完成
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                # 播放完毕后等待两秒
                time.sleep(2)

            else:
                print("识别失败，未得到结果")


if __name__ == "__main__":
    # OpenAI API　Key
    openai_api_key = ''
    # Baidu ID
    baidu_speech_appid = ''
    # Baidu API Key
    baidu_speech_api_key = ''
    # Baidu Speech API Key
    baidu_speech_secret_key = ''
    # Baidu Translate ID
    baidu_translate_appid = ''
    # Baidu Translate Key
    baidu_translate_secret_key = '' 

    chatbot = IntegratedChatbot(openai_api_key, baidu_speech_appid, baidu_speech_api_key, baidu_speech_secret_key, baidu_translate_appid, baidu_translate_secret_key)
    chatbot.recognize_and_chat()
