import openai

from api_key import api_key

openai.api_key = api_key

chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "문장의 감정을 판단해 행복, 즐거움, 불행, 감동, 호기심, 집착, 희망, 설렘, 기대, 원망, 분노, 증오, 짜증, 질투의 감정 중에서 적절한 단어로 표현해 줘. 감정을 잘 모르겠으면 모름이라 말해."
        },
        {
            "role": "user",
            "content": "오늘은 많은 것을 공부할 수 있을거야"
        },
        {
            "role": "assistant",
            "content": "희망"
        },
        {
            "role": "user",
            "content": "인프런 강의 너무 비싸"
        },
        {
            "role": "assistant",
            "content": "짜증"
        },
        {
            "role": "user",
            "content": "인프런이 세상에서 제일 최고야~"
        },
        {
            "role": "assistant",
            "content": "행복"
        },
        {
            "role": "user",
            "content": input()
        },
    ],
    max_tokens=100,
    temperature=1,
    n=1,
)

print(chat)

for choice in chat.choices:
    print("-----------------------")
    print(choice.message.content)
