import os

import openai
from dotenv import load_dotenv

load_dotenv()

# 발급받은 API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-3.5-turbo"

# 질문 작성하기
query = "Explain ChatGPT."

# 메시지 설정하기
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": query},
]
# messages = [
#     {"role": "system", "content": "You are an assistant to help with addition. You take two numbers and output the result of their addition."},
#     {"role": "user", "content": "3, 5"},
#     {"role": "assistant", "content": "3 + 5 = 8 입니다."},
#     {"role": "user", "content": "10과 13"},
# ]

# ChatGPT API 호출하기
response = openai.ChatCompletion.create(model=model, messages=messages)
answer = response["choices"][0]["message"]["content"]
print(answer)

# 내용 요약하기

# 영어 -> 한국어로 번역하기
