import openai

from api_key import api_key

openai.api_key = api_key

chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "GPT를 만든 회사 OpenAI에 대해 짧게 설명해 줘 "
        }
    ],
    max_tokens=2000,
    temperature=1,
    n=2,
)

print(chat)

for choice in chat.choices:
    print("-----------------------")
    print(choice.message)
    print(choice.message.content)
