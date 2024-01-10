import openai
from .llm import Llm
from .prompt_template import build

# https://platform.openai.com/docs/models/gpt-4

class GPT3_5(Llm):
    def set_api_key(self, key):
        openai.api_base = 'https://api.closeai-asia.com/v1' # 固定不变
        openai.api_key = key 

    def build_prompt(self, instruction, tools, examples):
        return build(instruction, tools, examples, False, False)

    def invoke(self, text, tools=None, errors=None):
        messages=[{"role": "system", "content": "You are a GIS domain expert and a helpful assistant."},
                      {"role": "user", "content": text}]
        if tools!=None:
            messages+=[{'role': 'assistant', 'content': tools}]
        if errors!=None:
            messages+=[{'role': 'user', 'content': errors}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages)
        content = response.choices[0].message.content
        from .base import predeal
        return predeal(content)
        

