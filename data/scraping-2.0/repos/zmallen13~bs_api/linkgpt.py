import os
import openai
import requests
from bs4 import BeautifulSoup
def linkget_response(prompt):
    r = requests.get(prompt)
    soup = BeautifulSoup(r.text, "html.parser")
    prompt = soup.text

    # Set your OpenAI API key here
    openai.api_key = open('API', 'r').read()
    chat_line ="You are a fake news detector. Please evaluate the following news article, and clearly and separately state sources. please keep it in a formated way with if its true or not first the"
            
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chat_line
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

        # Print the assistant's response
    responeMain = response['choices'][0]['message']['content']
    return responeMain


