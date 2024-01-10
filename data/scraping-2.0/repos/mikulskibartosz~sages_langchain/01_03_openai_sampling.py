# 1. Jak użyć OpenAI API w Pythonie?
## Generowanie wielu tekstów jednocześnie

import openai


if __name__ == '__main__':
    openai.api_key_path = 'api.key'

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Litwo, Ojczyzno moja!",
        max_tokens=100,
        temperature=0.5,
        n=5
    )

    for choice in response.choices:
        print(choice.text)
        print('-' * 80)