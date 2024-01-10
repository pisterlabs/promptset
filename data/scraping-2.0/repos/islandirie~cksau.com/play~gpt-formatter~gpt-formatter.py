from flask import Flask, render_template, request
import openai

app = Flask(__name__)
openai.api_key = 'your-api-key'

def format_job_posting(unformatted_text):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that formats job postings."},
        {"role": "user", "content": f"Format the following job posting:\n{unformatted_text}"}
    ]

    response = openai.Completion.create(
        engine="text-davinci-002",
        messages=conversation,
        max_tokens=150
    )

    assistant_reply = response['choices'][0]['message']['content']
    return assistant_reply

@app.route('/', methods=['GET', 'POST'])
def index():
    formatted_posting = ""

    if request.method == 'POST':
        unformatted_posting = request.form['unformatted_posting']
        formatted_posting = format_job_posting(unformatted_posting)

    return render_template('index.html', formatted_posting=formatted_posting)

if __name__ == '__main__':
    app.run(debug=True)
