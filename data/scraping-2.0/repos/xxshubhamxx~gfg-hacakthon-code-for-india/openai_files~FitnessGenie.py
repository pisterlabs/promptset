from flask import Flask, request
from dotenv import load_dotenv
import os
import openai
from datetime import datetime
import base64

load_dotenv()
openai.api_key = base64.b64decode(os.getenv('OPENAI_KEY')).decode("utf-8")

def ai_response(data, pred):
    prompt = "Suggest a workout plan for me. My health details are: " + data + " Please note that height is in cm and weight is present in kg. I also created a machine learning model trained using Google's AutoML to predict whether I am healthy or not. Here are the results: " + pred + " Don't rely on model's output as it has only about 90% accuracy. Suggest a detailed workout plan for me in about 200 words in list format."
    print("ChatGPT Prompt: ", prompt)
    response = openai.Completion.create(
        engine = 'text-davinci-003',
        prompt = prompt,
        temperature = 0.75,
        max_tokens = 300
    )
    print("ChatGPT Response: ", response)
    refine_response = str(response['choices'][0]['text'])
    paragraphs = refine_response.split('\n')
    html_response = ''
    for p in paragraphs:
        html_response += f'<p>{p}</p>'
    html_response.replace('•', '<br> •')
    return html_response

def billing_resp():
    html_resp = ''
    today = datetime.today().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bal = openai.api_requestor.APIRequestor().request("get", "/usage",{"date": today}, {"time": time}, {"timestamp": timestamp})
    resp = bal[0].data['data']
    # print("Billing Usage Respose: ",resp)    # Prints list of all the API calls made till the current time
    total_usage = sum([item['n_generated_tokens_total'] for item in resp])
    remaining_tokens = 900000 - total_usage
    print(f"Remaining credits: {remaining_tokens}")
    html_resp += f"<p>Remaining tokens: {remaining_tokens} (approx ${round(remaining_tokens*0.02/1000,2)} out of $18 ) </p>"
    return html_resp