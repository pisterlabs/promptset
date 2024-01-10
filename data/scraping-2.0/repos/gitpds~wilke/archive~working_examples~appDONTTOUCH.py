
from flask import Flask, request, session, redirect, url_for, render_template
from twilio.twiml.messaging_response import MessagingResponse
from pdsbot import ask
import openai
import os



model="text-davinci-003",

    
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
app.config['SECRET_KEY'] = '323434'

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        animal = request.form["animal"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(animal),
            temperature=0.6,
        )
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)


def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero!

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )



@app.route("/textpdsbot", methods=("GET", "POST"))
def textbot():
    incoming_msg = request.values.get('Body')
    # chat_log = session.get('chat_log')
    # answer = ask(incoming_msg, chat_log)
    answer = ask(incoming_msg)
    # session['chat_log'] = append_interaction_to_chat_log(incoming_msg,  answer, chat_log)
    msg = MessagingResponse()
    msg.message(answer)
    return str(msg)

if __name__ == "__main__":
    app.run(debug=True)
    