import os
import openai
from dotenv import load_dotenv


class GPTModel:
    def __init__(self):
        load_dotenv()
        # 발급받은 API 키 설정
        self.OPENAI_API_KEY = os.getenv("GPT_API_KEY")
        # openai API 키 인증
        openai.api_key = self.OPENAI_API_KEY

    # history는 이전 대화 내용이나 이전 문진폼 결과를 넣어주면 됨
    def edit_date(self, query, history=None):
        # 모델 - gpt-4-0613 or "gpt-3.5-turbo" 선택
        model = "gpt-3.5-turbo"
        # 메시지 설정하기
        messages = [
            {
                "role": "system",
                "content": "You need to help pre-process the data. You just need to calculate how old it is as of 2023. Just print out the data type: list of numbers.  For example, if it's 2010.04.27, you can return only 13. Data formats vary from 1997-06-26,19970626, 2012-07-25  준공, 2023.7.31, 23.7.31 준공, 23년 7월 준공, 1989년4월20일. If it's too weird like 미등기 or unknown, set it to -1",
            },
            {"role": "user", "content": query},
        ]
        if history != None:
            messages.append({"role": "assistant", "content": history})

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=model, messages=messages)

        return response["choices"][0]["message"]["content"]

    def make_ask(self, query, history=None):
        # 모델 - gpt-4-0613 or GPT 3.5 Turbo 선택
        model = "gpt-4-0613"
        prompt = "Speak in Korean. Your role is that of a doctor. The user enters his/her painful symptoms. All you have to do is question the patient about one or more additional things like a doctor.\
You're asking questions to get as much information as possible from the patient. You should only ask questions.\
Example 1\
User: My leg hurts.\
Doctor: Where exactly does it hurt?\
User: My ankle hurts.\
Doctor: How does it hurt?\
User: It stings.\
Doctor: When did it start hurting?\
User: Has it been hurting since yesterday?\
Doctor: How did you get sick?\
User: I fell.\
\
Example 2\
User: My stomach hurts.\
Doctor: Where specifically does your stomach hurt?\
User: My lower stomach hurts.\
Doctor: How does it hurt?\
User: It's rumbling.\
Doctor: When did it start?\
User: I've been sick since this morning.\
Doctor: What kind of food did you eat?\
User: I ate raw food.\
\
Example 3\
User: I have a sore throat.\
Doctor: Where and how does it hurt?\
User: It hurts when I swallow.\
Doctor: When did it start hurting?\
User: I woke up this morning with a really sore throat.\
Doctor: Do you have a fever?\
User: I have a little.\
"
        # 메시지 설정하기
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        if history != None:
            messages.append({"role": "assistant", "content": history})

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=model, messages=messages)

        return response["choices"][0]["message"]["content"]


# 사용 예시:
"""
gpt_model = GPTModel()
#format 부분 바꿔주기
answer = gpt_model.make_form("Recommended medical department:{0}"+"Symptoms entered by the patient:{1}").format("Dermatology", "Skin rash, itching, redness")
print(answer)
"""
