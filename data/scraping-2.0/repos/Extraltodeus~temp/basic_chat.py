import os
import sys
import openai
from time import sleep
from dotenv import load_dotenv
import json
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
def append_string_to_file(file_path: str, text: str):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(text + '\n')
def append_dicts_to_file(file_path,dicts):
    with open(file_path, 'a', encoding='utf-8') as file:
        for d in dicts:
            line = json.dumps(d, ensure_ascii=False)
            file.write(line + '\n')

# "gpt-3.5-turbo","gpt-4","gpt-4-32k"
class basic_chat(object):
    """docstring for basic_chat."""

    def __init__(self,system_prompt="",max_tokens=300,temperature=1,presence_penalty=0,model="gpt-3.5-turbo",name=""):
        super(basic_chat, self).__init__()
        self.system_prompt = system_prompt
        self.history = []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.name = name

    def append_to_history(self,role,content):
        self.history.append({"role":role,"content":content})

    def remove_prompt_by_role(self,role="system"):
        new_history = []
        for msg in self.history:
            if msg['role'] != role:
                new_history.append(msg)
        self.history = new_history

    def add_system_prompt(self):
        if self.system_prompt == "": return
        system_prompt_dicted = {"role":"system","content":self.system_prompt}
        self.history.insert(0,system_prompt_dicted)
        if len(self.history)>4:
            self.history.insert(len(self.history)-2,system_prompt_dicted)

    def generate_message(self):
        append_dicts_to_file("comm_log.txt",self.history)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.history,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            presence_penalty=self.presence_penalty
        )
        return response['choices'][0]["message"]["content"]

    def clear_history(self):
        self.history = []

    def communicate(self,message=""):
        self.remove_prompt_by_role()
        self.add_system_prompt()
        if message != "" :
            self.append_to_history('user',message)
        retries = 5
        for retry in range(retries):
            try:
                answer = self.generate_message()
            except Exception as e:
                print(e)
                if retry == retries-1:
                    print("max retries reached : breaking")
                    break
                sleep(5)
            else:
                self.append_to_history('assistant',answer)
                self.remove_prompt_by_role()
                break
        return self.history[-1]['content']
