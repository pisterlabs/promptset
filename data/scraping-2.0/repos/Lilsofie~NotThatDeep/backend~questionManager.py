import openai
import random
from api_secrets import API_KEY
from flask import jsonify

openai.api_key = API_KEY
qotd = ""
topics = {"clown": "Generate an embarassing who-is-most-likely-to question that can be answered with a name. Make sure the person who receives votes will regret it to the end of their lives.",
          "crown": "Generate a who-is-most-likely-to question that can be answered with a name. Make sure  the pereson who receives will feel proud of themselves."}

prompt = random.choices(list(topics.items()), weights = (2,1), k = 1)

def generate_qotd():
    global qotd
    qotd = openai.Completion.create(engine="text-davinci-003", prompt=prompt[0][1], max_tokens=30)["choices"][0]["text"]
    return prompt[0][0]


def get_qotd():
    return jsonify({'qs': qotd.strip()}), 200