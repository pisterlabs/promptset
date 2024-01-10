import openai

from config import API_CHATGPT

def chat_work(text: str) -> str:
    openai.api_key = API_CHATGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": 'Complete the task in English write the correct answer' + ' ' + text}],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']

