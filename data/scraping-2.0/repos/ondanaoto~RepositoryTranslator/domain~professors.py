from typing import Any
from abc import ABC, abstractmethod
from openai import OpenAI

import setting
from domain.prompt import *



class LLM(ABC):
    
    @abstractmethod
    def make_prompt(self, context: dict[str, Any]={}) -> Prompt:
        return NotImplementedError
        
    def response(self, context: dict[str, Any]) -> str:
        prompt = self.make_prompt(context)
        client = OpenAI(api_key=setting.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[message.to_dict() for message in prompt.messages],
            temperature=prompt.temperature
        )
        return response.choices[0].message.content
        

class LLMFileTranslator(LLM):
    temperature = 0.0
    
    def make_prompt(self, context: dict[str, Any]) -> Prompt:
        file_ext = context["file_extension"]
        target_language = context["target_language_code"]
        raw_text = context["raw_text"]
        
        system_prompt = Message(Role.SYSTEM, f"You are a perfect translator who translates {file_ext} files into {target_language}.")
        user_prompt = Message(Role.USER, f"In cases where the document is lengthy, it's not necessary to forcibly summarize the entire content. If the translation needs to be cut short due to the document's length, that's perfectly fine. Please translate the following {file_ext} file into {target_language} and return only the translation content. \n\n{raw_text}")
        
        return Prompt([system_prompt, user_prompt], self.temperature)