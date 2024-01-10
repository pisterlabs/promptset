from app.functions import getConfig

import openai

defaultMessage = [
    {
        "role": "system",
        "content": "Context: You are an AI voice assistant named amber.",
    }
]


def openaiResponseCompletion(newMessages):
    print([*defaultMessage, *newMessages])
    openai.api_key = getConfig["openaiKey"]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[*defaultMessage, *newMessages]
    )
    print(completion)
    return completion
