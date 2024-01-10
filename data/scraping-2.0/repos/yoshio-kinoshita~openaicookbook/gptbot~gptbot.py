from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import openai
import os
import json
import markdown
import logging

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
socketio = SocketIO(app)

# ファイルへのハンドラを作成します。
file_handler = logging.FileHandler("app.log")

# ハンドラにフォーマッタを設定します。
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# ハンドラをロガーに追加します。
app.logger.addHandler(file_handler)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/socket")
def socket():
    return render_template("socket.html")

@app.route("/api/chat", methods=["POST"])
def chat():

    req_data = request.get_json()

    user_input = req_data['user_input']
    app.logger.info(user_input)

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "あなたは優秀なプログラマで、質問に対してとてもわかりやすい回答ができます。正確な回答を得るため、不明瞭なところは質問者に確認してください。",
            },
            {"role": "user", "content": user_input},
        ],
    )

    chatbot_response = response["choices"][0]["message"]["content"]
    md = markdown.markdown(chatbot_response)

    app.logger.info(chatbot_response)

    return jsonify({"response": md})

@app.route('/chat', methods=['POST'])
def chunk_chat():
    user_input = request.form['user_input']
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ],
        max_tokens=100,
    )
    chunks = [chunk['value'] for chunk in response['choices'][0]['message']['content'] if chunk['type'] == 'chunk']
    return {'chunks': chunks}

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected'})

if __name__ == "__main__":
    app.logger.setLevel(logging.DEBUG)
    os.environ["FLASK_ENV"] = "development"
    app.run(host="0.0.0.0", port=5050, debug=True)
