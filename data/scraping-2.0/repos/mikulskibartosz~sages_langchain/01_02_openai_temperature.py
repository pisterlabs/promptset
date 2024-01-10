# 1. Jak użyć OpenAI API w Pythonie?
## Parametr "temperature" 0-2

import openai


if __name__ == '__main__':
    openai.api_key_path = 'api.key'

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Litwo, Ojczyzno moja!",
        max_tokens=200,
        temperature=0.8
    )
    print(response.choices[0].text)