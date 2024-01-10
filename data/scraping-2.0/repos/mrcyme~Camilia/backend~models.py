# models.py
from openai import OpenAI
import json

with open("./keys.json", 'r') as j:
    keys = json.loads(j.read())
    OPENAI_API_KEY = keys["OPENAI_API_KEY"]

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def get_response(self, prompt):
        raise NotImplementedError("Subclasses should implement this method.")


class OpenAIModel(Model):
    def __init__(self, model_name="gpt-3.5-turbo"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def get_response(self, conversation, max_tokens=4096):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=conversation,
            max_tokens=max_tokens,
        )
        return response.choices[0].message

def get_model(model_name):
    # Depending on your actual implementation, you might do more here.
    if model_name in ["gpt4", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4-vision-preview"]:
        return OpenAIModel(model_name)
