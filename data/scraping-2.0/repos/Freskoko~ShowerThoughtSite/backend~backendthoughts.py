import requests
import os
import openai
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
openai.api_key = apikey

def get_showerthought():

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", #model="gpt-4", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates unique and interesting shower thoughts. Do not mention showerthoughts in replies, simply provide them. You may make the showerthought about the shower sometimes"},
            {"role": "user", "content": "Generate a shower thought for me."}
        ],
        temperature=0.95,
        max_tokens=50,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
        
    return response.choices[0]["message"]["content"]


def get_image_from_text(text):

    prompt = text

    response = requests.post(
        'https://api.openai.com/v1/images/generations',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {apikey}'
        },
        json={
            'prompt': f"colourful, shower, inspiring, {prompt} in a shower",
            'n': 1,
            'size': '1024x1024'
        }
    )

    return( response.json()["data"][0]["url"] )

if __name__ == "__main__":
    
    thought = (get_showerthought())
    print(thought)

    print(get_image_from_text(thought))

