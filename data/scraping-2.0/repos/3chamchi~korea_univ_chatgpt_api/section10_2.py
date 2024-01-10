# Section10 - 2
# ChatGPT API 연동 openai 패키지 사용

# 1. API 연동 준비
import openai

openai.api_key = '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'
# 2. API 요청
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            'role': 'user',
            'content': '반가워 ChatGPT',
        }
    ]
)

# 3. 응답 값 확인
print(response.choices[0].message)  # openai의 클래스인 경우
print(response.choices[0].message["content"].decode())  # openai의 클래스인 경우
# print(response['choices'][0]['message'])  # dict 형태인 경우
