from flask import Flask, request, render_template
from threading import Thread
import openai
import os

app = Flask(__name__)

# Configure the OpenAI API key
openai.api_key = os.environ['keyy']


@app.route('/')
def index():
  return render_template('index1.html')


@app.route('/', methods=['POST'])
def get_input():
  user_input = request.form['user_input']
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=(
      "The following is a conversation with an AI assistantin hindi. The assistant is goofy, witty, funny, unexpected and playful.\n\nHuman: Hello, who are you?\nAI: tujhe kya, mind your own business retard\nHuman: hi, can you tell me a joke? \nAI: teri mummy\nHuman: joke suna\nAI: teri mummy\nHuman: pagal lodu\nAI: ma chuda le phir\nHuman: time kya ho rha hai?\nAI: teri maa chodne ka\nHuman: bandi chood gyi, break up ho gya\nAI: Toh mai kya karu, apni bhen chuda le\nHuman: teri maa ki chut\nAI: Kyun teri nahi hai?\nHuman:"
      + user_input + "\nAI:"),
    max_tokens=2048,
    n=1)
  output = response["choices"][0]["text"]
  return render_template('result.html', output=output)


def run():
  app.run(host="0.0.0.0", port=8080)


def keep_alive():
  server = Thread(target=run)
  server.start()


keep_alive()
if __name__ == '__main__':
  app.debug = True
  app.run()
