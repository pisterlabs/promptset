import openai
from src.core.util.core_util import get_secret_data


SECRET_DATA = get_secret_data()
openai.api_key = SECRET_DATA["OPENAI_API_KEY"]


def ask_chat_gpt(question : str):
    # TODO - 돈들어오면 주석 해제
    chat_gpt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
    )
    result = chat_gpt['choices'][0]['message']['content']
    # result = "answer"
    return result