# test03-chatbot.py

import openai

# 각자 OPENAI API KEY 지정 : 이 파일은 버전 관리에는 절대 넣지 마세요.
openai.api_key = "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"

# API KEY 설정에 오류가 있는 지 확인하기 위함
print("api_key :", repr(openai.api_key))

# 챗봇 응답 생성
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "당신은 지식이 풍부한 도우미입니다."},
        {"role": "user", "content": "세계에서 가장 큰 도시는 어디인가요?"},
    ],
)

print(response["choices"][0]["message"]["content"])