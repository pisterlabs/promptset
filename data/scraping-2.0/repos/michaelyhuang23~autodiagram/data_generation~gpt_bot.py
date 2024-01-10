import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY') 

print(os.getenv('OPENAI_API_KEY'))
class GPTBot:
    def __init__(self, instruction, auto_regressive=False):
       self.context = [{"role": "system", "content": instruction}]
       self.instruction = instruction
       self.auto_regressive = auto_regressive
    
    def gen_completion(self, message):
        message = {"role": "user", "content": message}
        self.context.append(message)
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=self.context
        )
        if not self.auto_regressive:
            self.context.pop()
        else:
            self.context.append(completion.choices[0]['message'])
        return completion.choices[0]['message']['content']
    
    def clear_context(self, new_instruction=None):
        if new_instruction is not None:
            self.instruction = new_instruction
        self.context = [{"role": "system", "content": self.instruction}]