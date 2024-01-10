import openai
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
openai.api_key = #your api key here

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    input_text = request.form["text"]
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=input_text,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.6,
    )
    output_text = response.choices[0].text.strip()
    return render_template("index.html", input_text=input_text, output_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)
