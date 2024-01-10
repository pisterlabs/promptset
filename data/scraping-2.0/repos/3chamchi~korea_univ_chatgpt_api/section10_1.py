import openai

# 1. API 연동을 위한 준비
openai.api_key = "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"

# 2. ChatGPT API 요청
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "반가워 ChatGPT"
        }
    ]
)

# 3. 응답 값 출력
print(response.json())
