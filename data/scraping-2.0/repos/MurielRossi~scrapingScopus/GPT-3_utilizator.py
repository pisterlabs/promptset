import json

import numpy
import openai
import pandas as pd
import os
import csv

import cleanFun

# open the file of the response
path = "answerResponse"
f = open('answerResponse\solutions.csv', 'w')

# creating the writer
writer = csv.writer(f)

df = pd.read_csv('answerResponse\myPDFCleaned.csv', sep=',')

print(df)
openai.api_key_path = 'api_key'
question = "\n What kind of algorithm is mentioned in the text?\n\n"
index = 0
for i in df:
    text = ""
    text = text + i
    index = index + 1
    print(text)
    text = text + question


    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text,
        temperature=0,
        max_tokens=30,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )

    str = response["choices"][0]['text']
    print(str)

    writer.writerow([str])

f.close()
