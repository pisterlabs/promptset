import os
import requests
import json
import openai

openai.api_key = '7f66b5661345437d80e661020d74a2c9'
openai.api_base = 'https://genaiusecases.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

deployment_name='GenAIDemo' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a code in Robot framework to open Facebook.com and input username and password. '
response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=1000)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
print(start_phrase+text)