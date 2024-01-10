import os

import openai
from flask import Flask, redirect, render_template, request, url_for, Response

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# load your username and password here
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

def check_auth(username, password):
    """This function is called to check if a username/password combination is valid."""
    return username == USERNAME and password == PASSWORD

def authenticate():
    """Sends a 401 response that enables basic auth."""
    return Response('You need to login.', 401, {'WWW-Authenticate': 'Basic realm="Login!"'})

def requires_auth(f):
    """A decorator to ensure the user is logged in."""
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route("/", methods=("GET", "POST"))
@requires_auth
def index():
    if request.method == "POST":
        question = request.form["question"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(question),
            temperature=0.6,
            max_tokens=250,
        )
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)

def generate_prompt(question):
    return """Answer the following question:
Question: {}
Answer:""".format(
        question.capitalize()
    )

if __name__ == "__main__":
    app.run(debug=True)
