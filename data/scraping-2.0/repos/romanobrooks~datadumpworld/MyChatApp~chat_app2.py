from flask import Flask, render_template, request
import openai
from api_secrets import OPENAI_API_KEY

app = Flask(__name__)

openai.api_key = OPENAI_API_KEY

def openai_chatbot(user_input):
    response = openai.Completion.create(
        engine="davinci",
        prompt=user_input,
        max_tokens=50
    )
    return response.choices[0].text.strip()

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        response = openai_chatbot(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html", user_input="", response="")

if __name__ == "__main__":
    app.run(debug=False)
