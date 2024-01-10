import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("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"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "Anda adalah sistem pendeteksi hoax, jawaban anda cukup menyatakan HOAX atau TIDAK HOAX"},
        {"role": "user", "content": "Jokowi adalah kodok"}
    ],
    model="gpt-3.5-turbo",
)
print(chat_completion.choices[0].message.content)