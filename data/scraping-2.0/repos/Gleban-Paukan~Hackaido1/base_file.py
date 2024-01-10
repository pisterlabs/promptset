import openai
import Hackaido
from TOKEN import token


def active_file():
    openai.api_key = token

    text, emotion = Hackaido.welcome()

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Rewrite text in emotion: {emotion}: {text}"}
        ]
    )

    print(completion.choices[0].message.content)
