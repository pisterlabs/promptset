from flask import Flask, render_template, request, jsonify
import time

import os
import openai

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form.get('user_message')
    bot_response = simulate_bot_response(user_message)
    
    time.sleep(0.5)  # Simulate a slight delays
    return jsonify({"bot_response": bot_response})

from datetime import datetime
def date():
    d=datetime.now().date()
    ar = datetime.now().isoweekday()
    dic = {'1': 'Monday', '2': 'Tuesday', '3': 'Wednesday', '4': 'Thursday', '5': 'Friday', '6': 'Saturday'}
    aa = str(ar)
    if aa in dic:
        day=dic[aa]
    return f'today is {day}, {d}'

# here="hey, I'm a basic chatbot!"
openai.api_key = os.getenv("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")


def simulate_bot_response(user_message):
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=user_message,
                        max_tokens=1000,
                        temperature=0.5)
    return response

if __name__ == '__main__':
    app.run(debug=True)
