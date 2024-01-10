import openai
import os

from pydantic import BaseModel
from typing import List
from ..io import to_text_file
from .prompt import get_stop_patterns

class ChatLLM(BaseModel):
    model: str = 'gpt-3.5-turbo'
    temperature: float = 0
    api_key:str= os.getenv("OPENAI_API") 

    def _generate(self, prompt: str):
        assert self.api_key != None, "please provide API key"

        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=get_stop_patterns()
        )
        return response.choices[0].message.content
    

    def generate(self,prompt: str, output_folder:str,num_loops:int):
        to_text_file(prompt,f"{output_folder}/step_{str(num_loops)}_prompt.txt")
        generated = self._generate(prompt)

        to_text_file(generated,f"{output_folder}/step_{str(num_loops)}_response.txt")
        return generated