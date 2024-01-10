from unittest.mock import Mock, patch

import openai
import os

class MockAiContext:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        
        self.log = []

    def get_input(self, name, op_type):
        return self.inputs[name]

    def get_secret(self, env_var_name):
        try:
            return os.environ[env_var_name]
        except KeyError:
            print(f"get_secret: Environment variable '{env_var_name}' is not set.")
            return None

    def set_output(self, name, value, operator):
        # Store the output data
        self.outputs[name] = value

    def add_to_log(self, message, color=None, save=False):
        # Store the log message
        self.log.append(message)

    def run_chat_completion(self, msgs=None, prompt=None):
        openai.api_key = self.get_secret('OPENAI_TOKEN')
    
        mn = 'gpt-3.5-turbo'
        temperature = 1.0
        
        if prompt is not None:
            msgs = [{"role": "user", "content": prompt}]
        
        completion = openai.ChatCompletion.create(
            model=mn,
            messages=msgs,
            temperature=(float(temperature) if temperature is not None else None)
        )

        res = completion.choices[0].message.content
        return res
        
    # Methods below are not supposed to be used by operators.
    def set_input(self, name, value):
        self.inputs[name] = value  
        
    def get_output(self, name):
        return self.outputs.get(name)


