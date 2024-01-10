from flask import Flask, request, jsonify
import openai
from flask_cors import CORS
import json
import os
#my funcs
import libList

myapp = Flask(__name__)
CORS(myapp)

@myapp.route("/chatGPT",methods=["POST"])
def gpt():
    APIKEY = ""
    openai.api_key = APIKEY
    date = request.json
    NowPrompt = date["prompt"]
    pluginName = date["pluginName"]
    print(NowPrompt,pluginName)
    # if pluginName not in dat.funcList or pluginName == "default":
    if pluginName == "default":
        print("default")
        path = "env.json"
        with open(path,"r",encoding="utf-8") as f:
            env = json.load(f)
            env = env["char"][0]
        name,age,tail,c,type = env["name"],env["age"],env["tail"],env["c"],env["type"]
        prompt = f"あなたの名前は{name}です。{type}という生き物で、{age}歳です。"
        t = []
        for i in tail:
            i = "'" + i + "'"
            t.append(i)
        s = "語尾には、" + "や".join(t) + "がつきます。"
        prompt += s
        prompt += c
        prompt = NowPrompt + prompt
    else:
        path = "C:\\Users\\xbzdc\\reazon-internship-backend-2\lib\developer\dog.json"
        with open(path,"r",encoding="utf-8") as f:
            env = json.load(f)
            env = env["char"][0]
        name,age,tail,c,type = env["name"],env["age"],env["tail"],env["c"],env["type"]
        prompt = f"あなたの名前は{name}です。{type}という生き物で、{age}歳です。"
        t = []
        for i in tail:
            i = "'" + i + "'"
            t.append(i)
        s = "語尾には、" + "や".join(t) + "がつきます。"
        prompt += s
        prompt += c
        prompt = NowPrompt + prompt
    #     print("else")
    #     prompt,img = dat.getFunc(pluginName)
    #     print(NowPrompt, dat.prompt)
    #     prompt = NowPrompt + dat.prompt

    # print(prompt)
    # print(img)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        #requency_penalty=0,
        #presence_penalty=0
    )
    # response = [{"prompt":response["choices"][0]["message"]["content"]}]
    return response["choices"][0]["message"]["content"]

@myapp.route("/upload",methods=["POST"])
def upload():
    dat = request.files["file"]
    dat.pluginName = dat.filename.replace(".py","")
    print(dat.filename, dat.pluginName)
    # os.mkdir("lib\\" + dat.filename)
    path = os.path.join(*["lib", dat.pluginName, "main.py"])
    print(path)
    # dat.save("lib\\" + dat.filename + "\\" + "main.py")
    dat.save(path)
    
    return "Received: " + dat.filename

if __name__ == "__main__":
    dat = libList.libHandler()
    myapp.run(port=8080,debug=True)