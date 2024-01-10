from flask import Flask, request, abort, Response
from flask import redirect
from flask_cors import CORS
from flask import render_template
import json

from hashlib import sha256
import os 
import openai
import re



SECRET = "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"
openai.api_key='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'

# -*- coding: utf-8 -*-
# Set up the model and prompt
model_engine = "text-davinci-003"



def split_into_sentences(text):
    
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"


    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences




def StoryTime(init_prompt):
    completion = openai.Completion.create(
    engine=model_engine,
    prompt=init_prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)
    init_response = completion.choices[0].text
    print(init_response)

    senz = split_into_sentences(init_response)
    print(senz)
    imgz = []
    for i in senz:
        imgz.append(openai.Image.create(
            prompt=i,
            n=1,
            size="1024x1024"
        ))
    print(imgz)

    url_list = []
    for item in imgz:
        url_list.append(item["data"][0]["url"])
        
    print(url_list)

    pages = [{'sentence': s, 'image': url} for s, url in zip(senz, url_list)]

    return pages






def getGPT(prompt):



    # init_prompt = input()
    
    init_prompt = prompt

    # while True:
    #   prompt_next = input()
    #   prompt_next_context = "I asked ChatGPT: \"" + init_prompt + "\" and ChatGPT replied: \"" + init_response + " \n" + prompt_next 
    #   # Generate a response
    #   completion = openai.Completion.create(
    #       engine=model_engine,
    #       prompt=prompt_next_context,
    #       max_tokens=1024,
    #       n=1,
    #       stop=None,
    #       temperature=0.1,
    #   )
    #   response_next = completion.choices[0].text
    #   print(response_next)


    # splits text into sentences accurately



    output = (StoryTime(init_prompt))
    
    return output




app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("MNMT.html")

@app.route("/about")
def about():
    return """
    <h1 style='color: red;'>I'm a red H1 heading!</h1>
    <p>This is a lovely little paragraph</p>
    <code>Flask is <em>awesome</em></code>"""







@app.route("/textgen", methods=['POST'])
def textgen():
    
    res = request.get_json()
    print (res)

    resraw = request.get_data()
    print (resraw)
    
    prompt = res['input']
    
    output = getGPT(prompt)

    status = {}
    status["server"] = "up"
    status["request"] = res 
    status['output'] = output

    statusjson = json.dumps(status)

    print(statusjson)

    js = "<html> <body>OK THIS WoRKS</body></html>"

    resp = Response(statusjson, status=200, mimetype='application/json')
    ##resp.headers['Link'] = 'http://google.com'

    return resp








@app.route("/dummyJson", methods=['GET', 'POST'])
def dummyJson():

    res = request.get_json()
    print (res)

    resraw = request.get_data()
    print (resraw)

##    args = request.args
##    form = request.form
##    values = request.values

##    print (args)
##    print (form)
##    print (values)

##    sres = request.form.to_dict()


    status = {}
    status["server"] = "up"
    status["request"] = res 

    statusjson = json.dumps(status)

    print(statusjson)

    js = "<html> <body>OK THIS WoRKS</body></html>"

    resp = Response(statusjson, status=200, mimetype='application/json')
    ##resp.headers['Link'] = 'http://google.com'

    return resp


if __name__ == '__main__':
    # app.run()
    # app.run(debug=True, host = '45.79.199.42', port = 8090)
    app.run(debug=True, host = 'localhost', port = 8090)  ##change hostname here
