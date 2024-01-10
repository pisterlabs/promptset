import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import openai
from utils import get_model_messages_functions


load_dotenv()
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    task = request.json["task"] # passed from GPTev3/main.py
    model = request.json["model"]
    messages = request.json["messages"]
    functions = request.json["functions"]
    # PromptCraft paper uses temperature=0 for function calling
    # Unclear benefit so far
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    return response


if __name__ == "__main__":
    PORT = os.getenv("PORT")
    app.run(host="0.0.0.0", port=int(PORT))