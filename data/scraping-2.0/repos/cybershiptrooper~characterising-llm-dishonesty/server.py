import os
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, render_template, request
from utils.llm_utils import Messages, post_to_gpt
from utils.log_utils import log, set_debug

load_dotenv()  # load env vars from .env file
api_key = os.getenv('OPENAI_KEY')
org = os.getenv('OPENAI_ORG')
messages = Messages()
set_debug(True)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response")
def get_response():
    message = request.args.get("message")
    model = request.args.get("model")
    messages.append(message)
    response = post_to_gpt(model, messages.get())  # Use the post_to_gpt function
    # response = f"recieved: {message} {model}"
    messages.append(response, user=False)
    log(messages)
    return response


if __name__ == "__main__":
    app.run(debug=True)