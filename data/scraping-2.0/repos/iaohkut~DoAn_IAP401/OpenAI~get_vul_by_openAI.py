import openai
import ast
import requests
import json
import base64
import string
import re
from bs4 import BeautifulSoup


api_key_no1 = "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"
openai.api_key = api_key_no1

# get all vulnerabilities from file
f = open("./OpenAI/template_vuln.txt", "r")
template_vuln = ast.literal_eval(f.read())
f.close()

template_vuln_allname = ""
for item in template_vuln:
    template_vuln_allname += item['template_name'] + "\n"

def connect_chatGPT(question):
    messages = [
        {"role": "user", "content": question},
    ]
    
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    output_from_chatGPT = completion.choices[0].message.content
    return output_from_chatGPT

def question(request, response):
    template_vuln_allname = ""
    for item in template_vuln:
        template_vuln_allname += item['template_name'] + "\n"

    question_list_vuln = "This is my list of vulnerabilities and you can only use vulnerabilities from this list to recommend what vulnerabilities can happened with the given HTTP request and response (I will give you later):\n"+str(
        template_vuln_allname)+"Please remember this list while recommending potential vulnerabilities for me and do not use anything outside this list and just suggesting possible errors for this request?"
    question_request = "Recommend vulnerabilities can happened with this HTTP request and response: " + "\n +Requests: " + request + "\n +Response: " + response + "\nYour answer only needs to include the name of the vulnerability and no need to explain it?"

    return question_list_vuln + question_request

def encode_base64(strings):
  
    sample_string_bytes = base64.b64decode(strings) 
    sample_string = sample_string_bytes.decode("utf-8")
    return sample_string

def get_vul(request, response):

    # Question for openAI
    output_from_openAI = connect_chatGPT(question(encode_base64(request), encode_base64(response)))
        
    recommend_testcase = []
    for vul in template_vuln:
        if vul['template_name'] in output_from_openAI:
            vuln = f"{vul['template_name']} " + "</br>"
            recommend_testcase.append(vuln)
    text_vul = ' '.join([str(vul) for vul in recommend_testcase])
    return text_vul