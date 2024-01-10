#Don't forget to install openai through pip
#Also, get your API key from the openai website and if youre not a part of the 
#GPT-4 beta, change the model from "gpt-4" to "gpt-3.5-turbo"

import openai
import os
import json
import requests
import webbrowser
import random 

def generate_dall_e_image(prompt):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai.api_key}'
    }
    data = {
        'model': 'image-alpha-001',
        'prompt': f'{prompt}',
        'num_images': 1,
        'size': '1024x1024'
    }
    url = 'https://api.openai.com/v1/images/generations'

    # Send the request to the DALL-E API
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()

    # Check if the response was successful
    if response.status_code == 200 and 'data' in response_json and len(response_json['data']) > 0:
        return response_json['data'][0]['url']
    else:
        return None


messages = [
    {"role": "system", "content": "Hello, what can I do for you today?"},
    {"role": "user", "content": "Can you put a tilde at the start of the response every time i ask you for a photo?"},
    {"role": "system", "content": "Yes, I can put a tilde in start of the response when you ask for a photo."},
    {"role": "user", "content": "When I ask you to create a photo, you will instead describe the photo in in detail and as photorealistic."},
    {"role": "system", "content": "Yes, I will do that."}
]
system_msg = "You are a creative assistant."
openai.api_key = "OPENAI API KEY HERE"

print("Say hello to your new assistant!")
while input != "quit()":
    temp = (random.randint(7,10)) * 0.1
    message = input("")
    messages.append( {"role": "user", "content": message} )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        top_p=temp,
        temperature=temp,
        frequency_penalty=0.0,
        presence_penalty=0.0 )
    
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")

    if "~" in reply:
        split_prompt = reply.split("~")[1].strip()
        image_url = generate_dall_e_image(split_prompt)
        print(image_url)
        
        if image_url != None:
            webbrowser.open(image_url)
            
            
