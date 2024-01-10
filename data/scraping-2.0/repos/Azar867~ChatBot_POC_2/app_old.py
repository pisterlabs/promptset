from flask import Flask, render_template, request, jsonify
import openai

openai.api_key = "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"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    chat_messages = [{'role': 'system', 'content': 'Welcome! How I can help you?'}, {'role': 'user', 'content': input}]
    return get_openai_response(chat_messages)

def get_openai_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
    )
    print("response :: ",response)
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    app.run()
