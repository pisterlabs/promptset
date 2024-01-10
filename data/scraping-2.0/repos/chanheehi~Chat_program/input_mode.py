import speech_recognition as sr
from pynput import keyboard
import msvcrt
import os
import openai
import pyttsx3


def Text2speech(name, text):   # name: 사용자 이름, text: 사용자가 입력한 텍스트
    if 'source' == text[0:6]:   # source로 시작하는 경우 아무 동작도 하지 않음
        pass
    else:
        text = text.split('\\')[0]
        engine = pyttsx3.init()
        engine.say(f'{name}님이 {text}')
        engine.setProperty('rate', 100)
        engine.setProperty('volume', 1)
        engine.runAndWait()

def Speech2text():
    r = sr.Recognizer()
    input_key = None
    print("Speak Anything[텍스트모드(ESC)/음성모드(Enter)]")
    with sr.Microphone() as source:
        audio = r.listen(source)

        if msvcrt.kbhit():  # 입력받을때까지 대기
            input_key = msvcrt.getch()
            
            if ord(input_key) == 13: # input_key가 Enter라면 음성 입력 받기
                try:
                    text = r.recognize_google(audio, language='ko')   # 하루ㅂ 50회 제한
                    print(text)
                except sr.UnknownValueError:    # 인식 실패
                    print('인식 실패') 
                except sr.RequestError as e:    # API Key 오류
                    print("요청 실패 : {0}".format(e))  
                return text
            elif ord(input_key) == 27: # ESC 입력시 텍스트모드 변환
                text = '/텍스트'
                return text
            
def ChatGPT():
    openai.api_key = os.environ["OPENAI_KEY"]
    messages = []

    print("Speak Anything[텍스트모드(ESC)/GPT모드(Enter)]")

    while True:
        if msvcrt.kbhit():  # 입력받을때까지 대기
            input_key = msvcrt.getch()
            
            if ord(input_key) == 13: # input_key가 Enter라면 ChatGPT 입력 받기
                    user_content = input("user : ")
                    messages.append({"role": "user", "content": f"{user_content}"})
                    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
                    assistant_content = completion.choices[0].message["content"].strip()
                    messages.append({"role": "assistant", "content": f"{assistant_content}"})   

                    print(f"GPT : {assistant_content}")
                    print("**GPT의 답변을 전송하시겠습니까 ?(Y/N)**")
                    
                    while True:
                        input_key = msvcrt.getch()
                        if ord(input_key) == 89 or ord(input_key) == 121 or ord(input_key) == 12635: # input_key가 Y라면 GPT 답변 전송 O
                            assistant_content = f"GPT 질문 : {user_content}'\n'GPT 답변 :  {assistant_content}"
                            return assistant_content
                        if ord(input_key) == 78 or ord(input_key) == 110 or ord(input_key) == 12636: # input_key가 N이라면 GPT 답변 전송 X
                            return ''
                        else:   # input_key가 Y가 아니라면 다시 입력받기
                            pass
                    
            elif ord(input_key) == 27: # ESC 입력시 텍스트모드 변환
                text = '/텍스트'
                return text
            else:
                 pass
        

