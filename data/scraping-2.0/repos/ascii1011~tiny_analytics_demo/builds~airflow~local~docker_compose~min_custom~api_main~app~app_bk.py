#!/opt/venv/bin/python
 
import os

import openai

from utils import download_image, generate_prompt
from flask import Flask, redirect, render_template, request, url_for
from flask_navigation import Navigation

app = Flask(__name__)
nav = Navigation(app)


@app.context_processor 
def inject_dict_for_all_templates():
    """ Build the navbar and pass this to Jinja for every route
    ref: https://stackoverflow.com/questions/71834254/flask-nav-navigation-alternative-for-python-3-10
    """
    # Build the Navigation Bar
    nav = [
        {"text": "Homepage", "url": url_for('index')},
        #{"text": "About Page", "url": url_for('about')},
        #{"text": "Pet Name", "url": url_for('pet')},
        #{"text": "Generate Images from keywords", "url": url_for('image_gen')},
        #{"text": "Dev chatbot", "url": url_for('devchatbot')},
        #{"text": "Generate Horror Stories", "url": url_for('horrorstorygenerator')},
        #{"text": "General chatbot", "url": url_for('chat')},
        #{"text": "Generate Study notes from keywords", "url": url_for('studynotes')},
        #{"text": "Generate Product name from keywords", "url": url_for('createproductname')},
    ]

    return dict(navbar = nav)


openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pet", methods=("GET", "POST"))
def pet():
    if request.method == "POST":
        animal = request.form["animal"]
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=generate_prompt(animal),
            temperature=0.6,
        )
        return redirect(url_for("pet", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)

@app.route("/image_gen", methods=("GET", "POST"))
def image_gen():
    """
    1 to 10 images 
    res: 256x256, 512x512, or 1024x1024 pixels"""
    if request.method == "POST":
        # keywords example: "a white siamese cat"
        keywords = request.form["keywords"]
        print(f"{keywords=}")

        response = openai.Image.create(
            prompt=keywords,
            n=2,
            size="1024x1024"
        )
        print(f"{len(response['data'])=}")
        result = ""
        for i in range(len(response['data'])):
            image_url = response['data'][i]['url']
            print(f"{i}){image_url=}")
            if result == "":
                result = download_image(image_url, prefix='_'.join(keywords.split()))
            else:
                result += " | " + download_image(image_url, prefix='_'.join(keywords.split()))
        print(f"{result=}")
        return redirect(url_for("image_gen", result=result))

    result = request.args.get("result")
    return render_template("image_gen.html", result=result)


@app.route("/devchatbot", methods=("GET", "POST"))
def devchatbot():
    """
    prompt example:
    "You: How do I combine arrays?\nJavaScript chatbot: You can use the concat() method.\nYou: How do you make an alert appear after 10 seconds?\nJavaScript chatbot"
    """
    if request.method == "POST":
        chat_input = request.form["chat_input"]
        
        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=chat_input,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["You:"]
        )
        print(f"{response=}")
        return redirect(url_for("devchatbot", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("devchatbot.html", result=result)

@app.route("/horrorstorygenerator", methods=("GET", "POST"))
def horrorstorygenerator():
    """
    prompt example:               
    Topic: wrench\nTwo-Sentence Horror Story:
    """
    if request.method == "POST":
        chat_input = request.form["chat_input"]
        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=chat_input,
            temperature=0.8,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0
        )
        print(f"{response=}")
        return redirect(url_for("horrorstorygenerator", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("horrorstorygenerator.html", result=result)


@app.route("/chat", methods=("GET", "POST"))
def chat():
    """
    prompt example:
    The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:
    """
    if request.method == "POST":
        chat_input = request.form["chat_input"]
        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=chat_input,
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" Human:", " AI:"]
        )

        print(f"{response=}")
        return redirect(url_for("chat", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("chat.html", result=result)

@app.route("/studynotes", methods=("GET", "POST"))
def studynotes():
    """
    prompt example:
    The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:
    """
    if request.method == "POST":
        chat_input = request.form["chat_input"]
        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=chat_input,
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" Human:", " AI:"]
        )

        print(f"{response=}")
        return redirect(url_for("studynotes", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("studynotes.html", result=result)


@app.route("/createproduct", methods=("GET", "POST"))
def createproduct():
    """
    prompt example:
    """
    if request.method == "POST":
        chat_input = request.form["chat_input"]
        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Product description: A home milkshake maker\nSeed words: fast, healthy, compact.\nProduct names: HomeShaker, Fit Shaker, QuickShake, Shake Maker\n\nProduct description: A pair of shoes that can fit any foot size.\nSeed words: adaptable, fit, omni-fit.",
            temperature=0.8,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        print(f"{response=}")
        return redirect(url_for("createproduct", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("createproduct.html", result=result)

