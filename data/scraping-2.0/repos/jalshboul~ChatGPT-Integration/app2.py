import openai
from flask import Flask, jsonify, request, render_template
import os 

# Set environment variables
keys = os.getenv('OPENAI_KEY')


api_key= keys

app = Flask(__name__)
openai.api_key = api_key

def generate_chat_response(user_input):
    model_engine = "davinci"
    prompt = f"User: {user_input}\nAI:"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    chat_response = response.choices[0].text.strip()
    return chat_response

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["user_input"]
    chat_response = generate_chat_response(user_input)
    response_data = {"response": chat_response}
    return jsonify(response_data)

@app.route("/")
@app.route("/index.html")
def index():
   return render_template("index.html")

if __name__ == "__main__":
    print(generate_chat_response("say hello in french"))
    
