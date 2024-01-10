from flask import Flask, request, jsonify
import openai


openai.api_key = "apikey"

app = Flask(__name__)

def chat_with_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
        )

    return response.choices[0].message.content.strip()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = chat_with_gpt(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response = chat_with_gpt(user_input)
        print("Chatbot: ", response)
