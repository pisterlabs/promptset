import openai
from flask import Flask
from flask import request, render_template
import urllib

app = Flask(__name__)
openai.api_key = "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"

@app.route ('/test')
def test():
    return ("hello")


@app.route ('/template')
def template():
    myprompt = request.args.get('prompt')
    response = openai.Completion.create(
      engine="davinci",
      prompt=myprompt,
      temperature=0.7,
      max_tokens=100,
      top_p=1.0,
      frequency_penalty=0.2,
      presence_penalty=0.0,
#       stop=["\n"]
    )

    responsetext = urllib.parse.quote (response.choices[0].text)
    return render_template ("testme.html", mytext = responsetext)

@app.route('/')
def start():
    output = "<center><h1>Meta-discipline Toolkit</h1>"
    output = output + "<font style='font-size:20px'>Enter search: <input style='font-size:20px' type=text id='myinput' size=80></input><br><button onclick='submit()'>Submit</button></center>"
    output = output + "\n<script>function submit() {window.location.href = './respond?prompt=' + document.getElementById('myinput').value}</script>"
    return (output)

@app.route('/respond')
def process_prompt():
    output = "<center><h1>Meta-discipline Toolkit</h1></center>"
    myprompt = request.args.get('prompt')
    output = output + "<h2>" + myprompt +"</h2><br><font style='font-size:20px'>"
    for t in range (0, 4):

        response = openai.Completion.create(
          engine="davinci",
          prompt=myprompt,
          temperature=0.7,
          max_tokens=100,
          top_p=1.0,
          frequency_penalty=0.2,
          presence_penalty=0.0,
    #       stop=["\n"]
        )
        output = output + "<div><input name='group1' type=radio id='" + str(t) + "'><label for='" + str(t) + "'><b>Option " + str(t+1) + ":</b>..." + response.choices[0].text + "</label></input></div><p>"
    output = output + "<center><h3>Select which option you are most interested in...<p>"
    output = output + "Why are you interested?: <input style='font-size:20px' type=text id='myinput' size=80></input><br><button onclick='submit()'>Submit</button></center>"
    output = output + "\n<script>function submit() {window.location.href = './respond?prompt=' + document.getElementById('myinput').value}</script>"

    return (output)

