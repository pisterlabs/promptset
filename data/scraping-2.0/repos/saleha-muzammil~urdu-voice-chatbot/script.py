from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

openai.api_key = "your key"

@app.route("/process_speech", methods=["POST"])
def process_speech():
    print("received speech")
    data = request.json
    transcript = data["speech"]

    messages = [
        {"role": "user", "content": transcript},
        {"role": "system", "content": "You are a chatbot who only replies in Urdu to the user."}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5
    )

    chatbot_response = response['choices'][0]['message']['content']
    print(chatbot_response)

    return jsonify({"response": chatbot_response})


if __name__ == "__main__":
    app.run(debug=True)
