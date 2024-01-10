
import openai
import os
import requests
import json

class Davinci_api:

    def __init__(self):
        self.prompt=""
        self.model="text-davinci-003"
        #self.model_engine="davinci"
        self.max_tokens=60
        self.temperature=0.7
        
        #self.openai.api_key = os.getenv("OPENAI_API_KEY")
    def get_response(self,api_key):
        try:
            openai.api_key = api_key
            response=openai.Completion.create(
   model=self.model,
    prompt=self.prompt,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
    top_p=1,
  frequency_penalty=0,
  presence_penalty=0
       )
            #return response.choices[0].text.strip()
            return response
        except :
            print("Connection Error ")

