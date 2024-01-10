# Section 17
# 감성 분석 프로젝트

# 1. API 요청 준비
import openai

openai.api_key = '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'

messages = [
    {
        'role': 'system',
        'content': 'You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative.'
    }
]

# 2. 사용자 메시지 생성
input_text = input('분석할 내용을 입력해주세요 >>> ')
user_message = {
    'role': 'user',
    'content': input_text
}

# messages = messages + [user_message]
messages.append(user_message)

# 3. API 요청
response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages)

# 4. API 응답 확인
print(response)
print(response.choices[0].message)
