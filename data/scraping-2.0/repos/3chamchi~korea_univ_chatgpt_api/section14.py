# Section 14
# 인터뷰 질문지 생성

# 1. API 요청 준비
# openai 가져오기
import openai

# 키 설정
openai.api_key = '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'

# user_text = input('진행하는 인터뷰에 대한 내용을 입력해주세요 : ')
question1 = input('1. 인터뷰의 목적 : ')
question2 = input('2. 인터뷰 대상자의 배경 : ')
question3 = input('3. 다루고 싶은 특정 주제 : ')
question4 = input('4. 대상 청중 : ')

# 2-1. 인터뷰 질문지 API 호출
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            'role': 'system',
            'content': '인터뷰 질문지 작성을 위해 작성을 보조하는 역할이야',
        },
        {
            'role': 'user',
            'content': '1. 인터뷰의 목적 : ' + question1
                       + '2. 인터뷰 대상자의 배경 : ' + question2
                       + '3. 다루고 싶은 특정 주제 : ' + question3
                       + '4. 대상 청중 : ' + question4
        }
    ]
)
# 2-2. API 확인
print(response.choices[0].message['content'])

# 3. 결과 확인
