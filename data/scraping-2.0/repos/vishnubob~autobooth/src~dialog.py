import random
from openai import OpenAI

client = OpenAI()
from . prompts.models import *
from . prompts import get_prompt, list_prompts

def get_random_prompt():
    prompts = list_prompts()
    prompt_name = random.choice(prompts)
    print(f"Running {prompt_name} prompt.")
    return get_prompt(prompt_name)

class PhotoboothDialog:
    def __init__(self, prompt=None):
        if prompt is None:
            prompt = get_random_prompt()
        self.messages = [{"role": "system", "content": prompt}]

    def parse_completion(self, completion):
        from pprint import pprint
        response = completion.choices[0].message
        pprint(response)
        self.messages.append(response)
        return AssistantMessage.parse_raw(response.content)

    def generate_response(self):
        #model="gpt-3.5-turbo-0613",
        #model="gpt-4-0613"
        model = "gpt-4-1106-preview"
        completion = client.chat.completions.create(
            model=model,
            temperature=1.0,
            messages=self.messages
        )
        return completion

    def get_response(self, people_count=None, message=None):
        user_message = UserMessage(people_count=people_count, message=message)
        self.messages.append({"role": "user", "content": user_message.json()})
        completion = self.generate_response()
        return self.parse_completion(completion)
