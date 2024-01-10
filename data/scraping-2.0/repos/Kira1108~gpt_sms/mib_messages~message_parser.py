from mib_messages.prompts import MessagePrompt
from mib_messages.config import get_settings

import openai
from dataclasses import dataclass
from pydantic import BaseModel

openai.api_key = get_settings().OPENAI_API_KEY

if openai.api_key is None:
    raise Exception("OPENAI_API_KEY not found.")


class OpenAIResponse(BaseModel):
    content:str
    prompt_tokens:int
    completion_tokens:int
    total_tokens:int


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    
    # test using fake response
    # return OpenAIResponse(
    #     content = "fake content",
    #     prompt_tokens=10, 
    #     completion_tokens=10,
    #     total_tokens=20)
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return OpenAIResponse(content = response.choices[0]['message']['content'],
                prompt_tokens = response.usage['prompt_tokens'],
                completion_tokens = response.usage['completion_tokens'],
                total_tokens = response.usage['total_tokens'])

@dataclass
class MessageParser:
    
    template_name:str = 'keywords'
    examples:list = None
    
    def __post_init__(self):
        self.prompt = MessagePrompt(template_name = self.template_name, examples= self.examples)
        
    def parse(self, message:str) -> OpenAIResponse:
        prompt_str = self.prompt.format(message = message)
        return get_completion(prompt_str)
    
    
    def __call__(self, message:str) -> str:
        return self.parse(message).content
    
    
if __name__ == "__main__":
    msgParser = MessageParser()
    print(msgParser.parse("<CashMonkey>: want more money, here's a lot of lenders, feel free to get").dict())
    
        