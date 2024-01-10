import os
import requests
import json
import openai
import random
from requests_oauthlib import OAuth1Session

openai.api_key = os.environ.get("OPENAPI_KEY")
openai.api_base = os.environ.get("OPENAPI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview' # this may change in the future

chatgpt_model_name=os.environ.get("MODEL_NAME") #This will correspond to the custom name you chose for your deployment when you deployed a model. 

def callGPT():
    topic = getTopic()
    question = f"What are the top trending keywords in {topic}. Please give keywords only '/' seperated and no extra text"
    print(question)
    # Send a completion call to generate an answer
    response = openai.ChatCompletion.create(
                    engine=chatgpt_model_name,
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": question}
                        ]
                    )
    keywords = response['choices'][0]['message']['content']
    keywordsList = keywords.split('/')
    print(keywordsList)
    emotion = getEmotion()
    tweetMessageString = f"Make a {emotion} tweet on " + random.choice(keywordsList)
    print(tweetMessageString)
    tweetResponse = openai.ChatCompletion.create(
                    engine=chatgpt_model_name,
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": tweetMessageString}
                        ]
                        ,max_tokens=200
                    )
    
    print(tweetResponse['choices'][0]['message']['content'])
    return tweetResponse['choices'][0]['message']['content']

def getEmotion():
    emotion = ['happy', 'sad', 'funny', 'excited', 'lazy', 'fear', 'anxiety', 'suprise']
    return random.choice(emotion)

def getTopic():
    topics = ['books', 'technology', 'movies', 'hollywood', 'tv shows', 'cartoons', 'games', 'life', 'socrates', 'philosophy', 'songs', 'art', 'artist', 'jokes']
    return random.choice(topics)