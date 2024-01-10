#%%


import openai
import os
import time
import sys
import json
from dataclasses import dataclass
from typing import List
from text_generation import generate, complete
openai.api_key = os.environ.get("OPEN_AI_FREE_API_KEY")
openai.api_base = 'https://api.pawan.krd/v1'

SYS_PROMPT = """You are a personal assistant. Your goal is to help me organize my life
                and make me more productive. I will message you things like tasks I have to do, ideas that come to my mind,
                projects I want to work on, and so on. I will also ask you questions about topics I am interested in 
                or that would be helpful for me to know, for instance, to accomplish a task I have to do. 
                You will have to organize all this information and help me make sense of it. For instance, you could
                create a to-do list for me, or a list of ideas I have had, or a list of projects I want to work on. You should also remember
                what I have told you and be able to answer questions about it."""


class GPT:
    def __init__(self, sys_prompt=SYS_PROMPT, model="gpt-3.5-turbo", temperature = 1):
        self._sys_messages = [{"role": "system", "content": sys_prompt}]
        self._messages = []
        self.response = ""
        self._model = model
        self._temperature = temperature
        
    def set_system(self, sys_prompt):
        self._sys_messages = [{"role": "system", "content": sys_prompt}]
    
    def add_system(self, sys_prompt):
        self._sys_messages.append({"role": "system", "content": sys_prompt})

    def completion(self, prompt, role = "user", chat=False):
        messages = self._sys_messages + [{"role": role, "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature, # this is the degree of randomness of the model's output
            max_tokens=1000,
        )
        self.response = response.choices[0].message["content"]
        if chat:
            self._messages = messages + [{"role": "assistant", "content": self.response}]
        return self.response

def chat(gpt):
    while True:
        prompt = input("You: ")
        if prompt == "exit":
            break

        print("Bot:", gpt.completion(prompt, chat=True))

GPT.chat = chat
#%%

if __name__ == "__main__":
    gpt = GPT()
    if len(sys.argv) > 1:
        gpt.chat()

# %%
