# Section 12
# 이미지 생성기

# 1. API 요청 준비
import openai

openai.api_key = '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'

input_text = input('생성하고자 하는 이미지의 내용을 입력해주세요 : ')

# 2. API 요청
response = openai.Image.create(
    prompt=input_text,
    size='256x256'
)

# 3. 응답 값 확인
print(response.to_dict())
