import openai
import pyttsx3
import speech_recognition as sr


r = sr.Recognizer()
engine = pyttsx3.init()

# 음성 속도를 300으로 설정
engine.setProperty('rate', 250)

openai.api_key = ""

messages = []

while True:
    with sr.Microphone() as source:
        print("말씀하세요:")
        audio = r.listen(source)

    text = r.recognize_google(audio, language='ko-KR')

    # user_content = input(f"user : {text}")

    messages.append({"role": "user", "content": f"{text}"}) # 사용자의 질문을 리스트에 추가

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    gpt_content = completion.choices[0].message["content"].strip() # 챗봇의 답변을 변수에 저장

    messages.append({"role": "assistant", "content": f"{gpt_content}"}) # 챗봇 답변을 리스트에 추가

    print(f"GPT : {gpt_content}") # 챗봇의 답변 출력

    engine.say(gpt_content)
    engine.runAndWait() # 답변이 끝날때 까지 대기
    engine.stop() # 대답 출력 중지

    # 입력 속도 개선
    # 예외처리
    # 텍스트와 음성 같이 받기