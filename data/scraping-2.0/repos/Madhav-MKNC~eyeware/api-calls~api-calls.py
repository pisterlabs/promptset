# openai API I/O

from dotenv import load_dotenv
from os import getenv

import openai

# loading api keys
load_dotenv()
openai.api_key = getenv('OPENAI_API_KEY')

class GPT:
    def __init__(self, id=0, model="gpt-3.5-turbo"):
        self.role = ['user', 'system'][id]
        self.model = model
        # role = 0 -> remembers conversation 
        # role = 1 -> stand-alone outputs
    
    def get(self, input):
        try:
            response = openai.ChatCompletion.create(
                model = self.model,
                messages = [
                    {"role": self.role, "content": input}
                ]
            )["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            print("[error]",str(e))
            return "GPT API call failed"


