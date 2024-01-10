from flask import Flask
from flask import request
from flask import Response
import requests
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
# Get BOT Token from telegram
token = os.environ.get("TELEGRAM_BOT_TOKEN")


def generate_answer(question):
    model_engine = "text-davinci-003"
    prompt = (f"Q: {question}\n"
              "A:")

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    answer = response.choices[0].text.strip()
    return answer


# To Get Chat ID and message which is sent by client
def message_parser(message):
    chat_id = message['message']['chat']['id']
    text = message['message']['text']
    print("Chat ID: ", chat_id)
    print("Message: ", text)
    return chat_id, text


# To send message using "SendMessage" API
def send_message_telegram(chat_id, text):
    url = f'https://api.telegram.org/bot6588304538:AAGkrEBX0T6X3mmQYV240nzaMDEGPF7P6jQ/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    response = requests.post(url, json=payload)
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        msg = request.get_json()
        chat_id, incoming_que = message_parser(msg)
        answer = generate_answer(incoming_que)
        send_message_telegram(chat_id, answer)
        return Response('ok', status=200)
    else:
        return "<h1>Something went wrong</h1>"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=5000)
