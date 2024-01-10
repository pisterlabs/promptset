import openai
from openai import OpenAI

def genereer_afbeelding(apik, invoer):
    client = OpenAI(api_key=apik)
    response = client.images.generate(
        model="dall-e-3",
        prompt=invoer,
        n=1,
        size="1024x1024",
        quality="standard"
    )
    return response.data[0].url

def simple_chat_completion(aikey, vraagtekst):
    client = OpenAI(api_key=aikey)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "system",
        "content": ""+vraagtekst
    }],
    temperature=0.5,
    max_tokens=2500
    )
    return response.choices[0].message.content




