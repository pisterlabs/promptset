import speech_recognition as sr
import os
from openai_connector import chatgpt


engine=sr.Recognizer()
mic = sr.Microphone(1)

def call():
    with mic as source:
        engine.adjust_for_ambient_noise(source)
        print('Bicara sekarang')
        rekaman = engine.listen(source)
        print('waktu habis')

        try:
            # BY GOOGLE #
            hasil=engine.recognize_google(rekaman,language='id-ID')
            # print(type(hasil))
            os.system('cls')
            print(f'Aku : {hasil}')
            print(f'Bot : {chatgpt(hasil).lstrip()}')
            # END BY GOOGLE #

        # Ketika Speech Recognition error #
        except sr.UnknownValueError:
            print('tidak terdeteksi')
        # Ketika Speech Recognition error #

        # Error yang lain #
        except Exception as e:
            print(e)
        # Error yang lain #
while(True):
    if input("Mulai voice recognition?(y/n)") == 'y':
        call()
    else:
        break
