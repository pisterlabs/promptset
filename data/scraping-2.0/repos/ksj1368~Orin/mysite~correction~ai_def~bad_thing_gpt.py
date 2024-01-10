import openai
from django.conf import settings

def bad_thing(text):
    # GPT-3.5 model 사용
    openai.api_key = settings.OPENAI_KEY
    model_name = 'gpt-3.5-turbo'

    # 프롬프트 설정
    prompt = text
    max_tokens = 300   # 생성할 최대 토큰 수
    temperature = 0.7  # 생성에 사용할 온도 값d

    # 프롬프트 엔지니어링
    modified_prompt = prompt + '이 자소서의 아쉬운 점을 100자 이내로 1개 알려줘' 

    # GPT 모델에 요청하여 출력 생성
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': modified_prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )

    # 생성된 출력 확인
    output = response.choices[0].message.content.strip()

    return output