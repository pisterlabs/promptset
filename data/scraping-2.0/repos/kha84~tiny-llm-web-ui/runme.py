#!/bin/env python
import argparse
import datetime
from flask import Flask, request, session, render_template, redirect, url_for
import openai

# Model to use. Possible values are: 'dummy', 'openai'
MODEL = 'dummy'


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--api_key", help="OpenAI API key")
parser.add_argument("--username", help="Accepted username")
parser.add_argument("--password", help="Accepted password")
args = parser.parse_args()
if not args.api_key:
    args.api_key = input("Enter API key:")
if not args.username:
    args.username = input("Enter username:")
if not args.password:
    args.password = input("Enter password:")

# Set up OpenAI API key
openai.api_key = args.api_key

if not args.api_key:
    print("python "+__file__+" --api_key YOUR_API_KEY --username ACCEPTED_USERNAME --password ACCEPTED_PASSWORD")
    exit()

app = Flask(__name__)
app.secret_key = 'your secret key'

glob_v = 0

def get_response(model, question):
    if (model == 'openai'):
        # Send question to ChatGPT and get response
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
        )
        answer = response['choices'][0]['message']['content']
        return answer
    if (model == 'dummy'):
        return 'He he he'

@app.route('/',  methods=['GET', 'POST'])
def root():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=args.username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check user credentials here
        if request.form['username'] == args.username and request.form['password'] == args.password:
            session['logged_in'] = True
            return redirect(url_for('root'))
    return render_template('login.html')

@app.route('/chat', methods=['GET', 'POST'])
def ask():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Get question from form data
        question = request.form['user_message']
        # Get IP address
        ip_address = request.remote_addr
        # Log IP address, timestamp, and question
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'{timestamp} | IP: {ip_address} | Question: {question}'
        with open('logfile.txt', 'a') as f:
            f.write(log_message + '\n')
        answer = get_response(MODEL, question)
        # Log the response
        log_message = f'{timestamp} | IP: {ip_address} | Answer: {answer}'
        with open('logfile.txt', 'a') as f:
            f.write(log_message + '\n')
        return {'user_message': question, 'bot_message': answer}

if __name__ == '__main__':
#    app.run()
    app.run(host='0.0.0.0')
