import os
import openai
from langchain.prompts import PromptTemplate
from typing import Dict

# SYN (hyperonomy)
# SEM (nn=composition; sv=transitivity; others=causality)

openai.api_key = "sk-"

class PromptTemplate:

    def __init__(self, prompt_template):
        self.prompt_template = prompt_template.replace("_", "{}")
        
    def format(self, *args):
        return self.prompt_template.format(*args)

class Suskind_LLM_Implementation:

    def __init__(self, builder, prompt_template):

        self.model_name = builder.model_name
        self.temperature = builder.temperature
        self.max_tokens = builder.max_tokens
        
        self.prompt_template = PromptTemplate(prompt_template)

        self.extract = builder.extract

        self.context_prompt = ''

    def set_context(self, context_prompt: str):
        self.context_prompt = context_prompt

    def send(self, *args):

        messages = [{"role": "user", "content": self.prompt_template.format(*args)}]
        if self.context_prompt:
            messages.insert(0, {"role": "system", "content": self.context_prompt})

        response = openai.ChatCompletion.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages
        )

        if self.extract:
            response = response['choices'][0]['message']['content']
        return response

class LLMBuilder:

    def __init__(self, max_tokens: int, temperature: float):

        self.model_name = 'gpt-3.5-turbo-16k'  # Default model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = ''

        self.extract = False

    def set_model_name(self, model_name: str):
        self.model_name = model_name
        return self
    
    def set_extract(self, extract: bool):
        self.extract = extract
        return self

    def build(self, prompt_template: str):
        llm = Suskind_LLM_Implementation(self, prompt_template)
        return llm

# Dictionary of models for easy access and scalability
MODELS: Dict[int, str] = {
    1: 'gpt-3.5-turbo-16k',
    2: 'gpt-3.5-turbo-1106',
    3: 'babbage-002',
    4: 'davinci-002'
}

# Example usage
def main():

    builder = LLMBuilder(max_tokens=2, temperature=0.0)
    builder.set_model_name(MODELS[2])
    builder.set_extract(True)

    # SEPARAR COSITAS

    prompt_models = {

        'synset' : ('("_"/"_") Which is more concrete? [1st/2nd]',
                    'Output example. "Bat"->"Tool-Mammal"'),

        'semset' : ('("_"/"_") Which scope is higher? [1st/2nd]',
                    'Just answer with "1st"/"2nd"'),

        'equivalence' : ('"_" is "_" [Y/N?',
                         ''),

    }

    choice_model = 'synset'
    choice_data = 'data'

    data = import_args(f'{data}.txt').random(n=3)    
    model = prompt_models[choice_model]
        
    for _ in data:
        print(f'Entry: {_}')
        response = model.send(*_[:2])
        print(f'Response: {response}')
        print(f'Expected: {_[-1]}')
        print('')

if __name__ == "__main__":
    main()
