"""
This is a sample flask app, to test a basic web service
"""

# pip install flask --quiet

import openai
from flask import Flask, request, render_template
import json
# create an app object of type class Flask
openai.api_key = "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"


def get_suggestions(prompt):
    messages = [{"role":"system","content":"You are an expert recruter, having 10+ years of experince in creating clear, precise and proper job description. You have knowledge of all technology and non tech roles in industry, you are well aware of skills and responsibilities for new jobs in market."},
                {"role":"user","content":prompt}]
    response = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature = 1.5,
    )
    return response['choices'][0].message.content


app = Flask(__name__)
# route the app to a URL to accept a GET request
@app.route("/",methods=['GET','POST'])
def fun1():
    return render_template("index.html")

# route the app to a URL to accept a GET request
@app.route("/suggest",methods=['GET','POST'])
def fun2():
    data = dict(request.form)
    print(data['desg'])

    # fetching skills
    prompt = f"Provide detailed list of skills for the job role {data['desg']}, only provide comma separated skills, technologies, tools. no explanation"
    skills = get_suggestions(prompt)

    # fetching job description
    prompt = f"Provide a detailed job description with components such as Job Summary, Qualifications, Responsibilities for the job role f{data['desg']}, Be clear and precise, provide bullet texts whereever applicable"
    jobdesc = get_suggestions(prompt)

    return render_template("index.html",job_desc=jobdesc,skills=skills,desg=data['desg'])

@app.route("/submit",methods=['GET','POST'])
def fun3():
    data = dict(request.form)
    print(data)
    return "Hello World from ChatGPT!!"

if __name__=="__main__":
    app.run(debug=True)