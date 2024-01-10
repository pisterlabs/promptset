import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_TOKEN')

class Gpt_API:
    def __init__(self,prompt):
        self.prompt = prompt

    def get_result(self):
        completion = openai.Completion.create(engine="text-davinci-002", prompt=self.prompt, max_tokens=2048, n=1, stop=None, temperature=0.5)
        #print(completion)
        print(self.prompt)
        text = completion.choices[0].text
        return text
