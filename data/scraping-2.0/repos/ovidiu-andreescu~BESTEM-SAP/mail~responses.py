import openai

openai.api_key = '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'
messages = []

def get_response(vect: str) -> str:
    p_message = f'Scrie un exemplu de mesaj trimis unui client de catre o companie trimis pentru a ii sugera sa cumpere unul sau mai multe iteme, enumerate aici:{vect}'

    messages.append({"role": "user", "content": p_message})
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )

    reply = chat.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})
    return reply
