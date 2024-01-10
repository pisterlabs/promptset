import openai

from api_key import api_key

openai.api_key = api_key

completion = openai.Completion.create(
    engine="text-davinci-003",
    prompt="GPT를 만든 회사 OpenAI에 대해 짧게 설명해 줘.",
    max_tokens=2000,
    temperature=0,
    n=2,
)

print(completion)

for choice in completion.choices:
    print("-----------------------")
    print(choice.text)
