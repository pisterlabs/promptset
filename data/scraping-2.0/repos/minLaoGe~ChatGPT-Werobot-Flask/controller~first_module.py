import openai
from flask import Flask, redirect, render_template, request, url_for, jsonify, Blueprint

from utils.globalConst import _const
from utils.requestToGPT import sendToGPT

import uuid

blueprint = Blueprint('products', __name__, url_prefix='/api')

const = _const.instance()
openaiAPi = const.OPENAI_APPID


@blueprint.route("/pet", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        animal = request.form["animal"]
        prompt = generate_prompt(animal)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.6,
        )
        return redirect(url_for("pet", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("pet.html", result=result)


@blueprint.route("/", methods=("GET", "POST"))
def index3():
    if request.method == "POST":
        abc = request.json['yourWant']
        text = """{}""".format(abc)
        result = sendToGPT(text)
        result = result.replace(".", ".<br/>")
        return jsonify({'result': result, 'error': None})
    result = request.args.get("result")
    return render_template("chatWithGpt.html", result=result)


@blueprint.route("/Traslation", methods=("GET", "POST"))
def index1():
    print("进入了/Translation")
    if request.method == "POST":
        text = request.json['text']
        lang = request.json['lang']
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt_TransLations(lang, text),
            temperature=0.6,
        )
        result = response.choices[0].text;
        return jsonify({'result': result, 'error': None})

    result = request.args.get("result")
    return render_template("index.html", result=result)


def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.
Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )


def generate_prompt_TransLations(lang, animal):
    return """Translation it to {}.
Animal: {}
Names:""".format(
        lang, animal.capitalize()
    )
def generate_uuid():
    return str(uuid.uuid1())
