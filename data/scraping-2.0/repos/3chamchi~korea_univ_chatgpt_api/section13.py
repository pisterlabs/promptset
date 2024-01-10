# Section 13
# 번역, 이미지 생성기

# 1. API 요청 준비
import openai  # pip install openai

openai.api_key = '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'

chat_text = input('생성할 이미지의 설명을 입력해주세요 : ')
# 2-1. 번역 API 요청
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            'role': 'system',  # system, user, assistant
            'content': '한국어를 영어로 번역해줘',
        },
        {
            'role': 'user',  # system, user, assistant
            'content': chat_text,
        },
    ]
)

# 2-2. 번역 API 응답 확인
# print(response.to_dict())
image_text = response.choices[0].message['content']
# print(image_text)
# 3-1. 이미지 생성 API 요청
image_response = openai.Image.create(
    prompt=image_text,
    size='256x256'
)
# 3-2. 이미지 생성 API 응답 확인
print(image_response)

# 4. 결과 출력
