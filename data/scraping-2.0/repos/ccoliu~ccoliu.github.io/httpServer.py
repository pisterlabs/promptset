# server.py
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
client = OpenAI()



@app.route("/", methods=["GET"])
def index():
    return "Hello, this is the home page!"


@app.route("/process_code", methods=["POST"])
def process_code():
    try:
        data = request.get_json()
        code = data.get("code", "")

        # 在這裡進行後端處理，這裡只是一個示例，你可以插入你的處理邏輯
        result = f"{code}"
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=150,
            messages=[
                {
                    "role": "system",
                    "content": "You are a teaching assistant, skilled in giving student's program advice about how they can  improve thier program without changing the required result",
                },
                {
                    "role": "user",
                    "content": code,
                }
            ],
        )
        print ("test")
        result = f"{completion.choices[0].message.content}"
        # 返回處理結果到前端
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)