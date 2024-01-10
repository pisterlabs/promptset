import openai
from models import generator
from datetime import datetime

# TODO: Replace with your API key or continue using ####
openai.api_key = '#####'

last_api_call = None

## Do not change this function
def get_response(messages):
  global last_api_call

  if last_api_call is not None:
    time_since_last_call = (datetime.now() - last_api_call).total_seconds()
    if time_since_last_call < 20:
      raise Exception("The chat requests must be at least 20 seconds apart.")
    
  last_api_call = datetime.now()

  if "###" not in openai.api_key:
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages) 
    return completion.choices[0].message.content
  else:
    last_message = messages[-1]
    response = generator(last_message["content"])
    return response[0]["generated_text"]
  