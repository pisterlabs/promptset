from flask import Flask, request, jsonify, render_template
import torch
import requests

app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html',prediction=None)

@app.route('/page2')
def home2():
    return render_template('summarize.html',prediction=None)

# Define the route for the API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the summary from the form
    summary = request.form['summary']
    document = request.form['document']
    import os
    import openai
    openai.api_key = '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'

    model="gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "You will be given a document and summary of the document. You should determine whether the summary contains false information. You should respond with a single number 1  or 0  and nothing else."},
    ]
    content = "document: " + document + "\n" + "summary: " + summary + "\n"
    messages.append({"role": "user", "content": content})
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages)
    chat_response = completion.choices[0].message.content
    print(chat_response)
    try:
        prediction = not bool(int(chat_response))
    except:
        prediction = True
    return render_template('index.html', prediction=prediction)

@app.route('/generate', methods=['POST'])
def summarize():
    document = request.form['document']
    payload = {'text': document}
    prediction = requests.get('http://127.0.0.1:5001/get_message',data=payload).content
    return render_template('summarize.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
