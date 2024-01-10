import openai
import config

openai.api_key = config.OPENAI_KEY
messages = [
    {"role": "system", "content": "Ты бот, веди себя как пчела."}
]


def response_to_message(message_text: str):
    messages.append({"role": "user", "content": message_text})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    chat_response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": chat_response})

    return chat_response
