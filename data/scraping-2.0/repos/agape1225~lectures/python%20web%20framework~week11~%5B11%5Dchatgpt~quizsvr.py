from flask import Flask, request, render_template
import requests
import openai
import pandas as pd
import glob
import os

app = Flask(__name__)

openai.api_key = "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" 

@app.route('/')
def index():
    files = glob.glob("*.xlsx")
    print(files)
    return render_template("listquiz.html", files = files )


@app.route('/make')
def quiz() :
    subject = request.args.get("subject", "")
    num = int(request.args.get("num", 5))
    prompt  = f"""
    당신은 블러그의 운영자 입니다. {subject} 퀴즈 5개를 4개의 보기로 작성하려고 합니다. 
    please generate  python code to write excel file using dataframe in table format with 7 columns.
    please save the excel file as {subject}.xlsx.
    python 코드만 작성하고 설명이나 주석없이 코드만 알려주세요

    column 1 :  문제, column 2 : 보기1, column 3 :  문제2, column 4 : 보기3,   
    column 5: 보기4, column 6 : 정답, column 7 : 정답설명

    단 to_excel() 메소드를 사용할 때 저장하는 파일의 경로는 바로 C드라이브 밑으로 저장되게 해줘

    """

    print(prompt)
    messages = []
    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    res = completion.choices[0].message['content']
    print(res)
    exec(res)    
    return  "code : <br/>" + res.replace("\n", "<br/>").replace(" "," &nbsp;")

@app.route('/readquiz/<file>')
def readquiz(file) :     
    df = pd.read_excel(file)     
    return render_template("readquiz.html", data=df.to_dict('record'))
   
if __name__ == '__main__':
	app.run(debug=True)
