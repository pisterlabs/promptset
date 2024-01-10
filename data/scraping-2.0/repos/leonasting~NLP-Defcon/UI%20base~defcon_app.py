# -*- coding: utf-8 -*-
"""
Created on  Oct 27 15:02:16 2022

@author: NLP Devcon
"""
from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
import openai
import urllib.parse
import requests
import time

openai.api_key = "xxxxxx"
MAX_RETRIES = 25
api_token = "xxxx"
headers = {"Authorization": "Bearer " + api_token}

@app.route("/")
def home():
    return render_template("main.html")

@app.route("/main", methods=["POST", "GET"])
def main():

    if request.method == "POST":

        contract= request.form["Contract"]
        question= request.form["question"]
        option= request.form["option"]
        model = request.form["models"]
        print("Contract:",contract)
        print("question:",question)
        print("Model", model)
        if option=="Fetch Answers" and model == "gpt":
            print("Query:","Write questions based on the text below\n\nText:"+contract+"\n\nQuestions:\n"+question+"\n\nAnswer:\n1.")
            response = openai.Completion.create(
                                                engine="davinci-instruct-beta",
                                                prompt=f"Get answer based on the question\n\nText:{contract}\n\nQuestions:\n{question}\n\nAnswer:\n1.",
                                                temperature=0,
                                                max_tokens=257,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0
                                                )
            ans =  response['choices'][0]['text']
            print("response",response)
            print("ans",ans)
            return render_template("label.html",answer=ans)#,table2=df2.to_html(classes='data'),url ='./static/images/new_plot2.png')
        elif option =="Fetch Answers" and model == "roberta":
            retries = 0
            query = dict(question=question, context=contract)
            model_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
            while retries < MAX_RETRIES:
                retries += 1
                r = requests.post(model_url, json=query, headers=headers)
                if r.status_code == 503:
                # We'll retry
                # If running under asyncio, be sure to use
                # `await asyncio.sleep(1)` instead.
                #logger.info("Model is currently loading")
                    time.sleep(1)
            ans = r.text
            return render_template("label.html",answer=ans)
        elif option =="Fetch Answers" and model == "distilbert":
            retries = 0
            query = dict(question=question, context=contract)
            model_url = "https://api-inference.huggingface.co/models/distilbert-base-cased-distilled-squad"
            while retries < MAX_RETRIES:
                retries += 1
                r = requests.post(model_url, json=query, headers=headers)
                if r.status_code == 503:
                # We'll retry
                # If running under asyncio, be sure to use
                # `await asyncio.sleep(1)` instead.
                #logger.info("Model is currently loading")
                    time.sleep(1)
            ans = r.text
            return render_template("label.html",answer=ans)
        elif option =="Fetch Answers" and model == "bert":
            retries = 0
            query = dict(question=question, context=contract)
            model_url = "https://api-inference.huggingface.co/models/deepset/bert-base-cased-squad2"
            while retries < MAX_RETRIES:
                retries += 1
                r = requests.post(model_url, json=query, headers=headers)
                if r.status_code == 503:
                # We'll retry
                # If running under asyncio, be sure to use
                # `await asyncio.sleep(1)` instead.
                #logger.info("Model is currently loading")
                    time.sleep(1)
            ans = r.text
            return render_template("label.html",answer=ans)
    else:
        return render_template("main.html")
if __name__ == "__main__":
    app.run(debug=True)
