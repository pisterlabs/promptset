from flask import Flask, request, jsonify, render_template
import openai

app = Flask(__name__)

openai.api_key = "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"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_openai():
    query = request.json.get('query')
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        max_tokens=150
    )
    return jsonify(response.choices[0].text.strip())

if __name__ == '__main__':
    app.run(debug=True)
