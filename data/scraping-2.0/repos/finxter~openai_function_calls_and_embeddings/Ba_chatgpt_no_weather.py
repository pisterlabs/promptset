from decouple import config
import openai

openai.api_key = config("CHATGPT_API_KEY")


def ask_chat_gpt(query):
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.4,
        messages=[
            {"role": "user", "content": query},
        ],
    )
    return result["choices"][0]["message"]["content"]


print(ask_chat_gpt("What is the weather in Seoul?"))
