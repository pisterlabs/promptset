from abc import ABC, abstractmethod
import numpy as np
import os
import openai
import requests
import json
import logging
from typing import List, Union, Optional, Dict
import scipy

class LLMWrapper(ABC):

    def __init__(self):
        self.params = {
            "max_new_tokens": lambda x : int(x),
            "temperature": lambda x: float(x),
            "skip_special_tokens": lambda x : bool(x)
        }

    @abstractmethod
    def generate_response(self, 
        input_str : str, 
        max_new_token: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> str:
        pass

    def cond_log_prob(self, inputs : str, targets: List[str], args : Dict) -> List[float]:
        raise Exception("not implemented")
    
    def single_cond_log_prob(self, input : str, target: str) -> float:
        raise Exception("not implemented")

    def clean_args(self, args):
        # remove args that are not defined in self.params
        valid_keys = filter(lambda x : x in self.params.keys(), args.keys())
        args = {key: args[key] for key in valid_keys}
        # get args
        args = {key: self.params[key](value) for key, value in args.items()}
        return args


class DummyLLM(LLMWrapper):

    def __init__(self):
        LLMWrapper.__init__(self)
        self.responses = [
            "Hello, I am the Dummy LLM.",
            "I say random stuff.",
            "I am the smartest LLM of all."
        ]

    def generate_response(self, input_str : str, args):
        args = self.clean_args(args)
        response = np.random.choice(self.responses)
        response += "\n\nYour input was: " + input_str 
        return response

class OpenAIBase(LLMWrapper):
     
    def __init__(self, app):
        LLMWrapper.__init__(self)
        self.model = app.get_args().model

    def set_openai_args(self, args):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        openai_args = {
                "model": self.model
            }
        
        if "max_new_tokens" in args.keys():
            openai_args["max_tokens"] = int(args["max_new_tokens"])

        if "temperature" in args.keys():
            openai_args["temperature"] = float(args["temperature"])

        return openai_args

class OpenAICompletion(OpenAIBase):

    def __init__(self, app):
        OpenAIBase.__init__(self, app)

    def generate_response(self, input_str : str, args):
        
        openai_args = self.set_openai_args(args)
        openai_args["prompt"] = input_str

        response = openai.Completion.create(**openai_args)
        return response["choices"][0]["text"]

class OpenAIChatCompletion(OpenAIBase):

    def __init__(self, app):
        OpenAIBase.__init__(self, app)

    def generate_response(self, input_str : str, args):

        openai_args = self.set_openai_args(args)
        openai_args["messages"] = [ {'role': 'user', 'content': input_str} ]

        response = openai.ChatCompletion.create(**openai_args)
        return response["choices"][0]['message']['content']

class RemoteHTTPLLM(LLMWrapper):

    def __init__(self, api_url):
        self.api_url = api_url

    def generate_response(self, input_str : str, args):
        doc = {"doc": input_str}
        x = requests.post(self.api_url, json = doc)
        return x.text
    
class OPENGPTX(LLMWrapper):

    def __init__(self):
        LLMWrapper.__init__(self)
        self.lastResponse = None

    def generate_response(self, input_str : str, args):

        headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        }
        
        post_data = {
            "inputs": input_str,
            "seed": 1,
            "parameters": {
                "sample_or_greedy": "sample",
                "max_new_tokens": args["max_new_tokens"]
            },
            "last_response": self.lastResponse,
        }

        x = requests.post("https://opengptx.dfki.de/generate", headers=headers, data = json.dumps(post_data))
        self.lastResponse = x.json()["output_text"]
        return (x.json()["output_text"])
