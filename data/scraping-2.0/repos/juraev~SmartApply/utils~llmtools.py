import os
from abc import ABC, abstractmethod

import openai
from openai import OpenAI

class LLMInterface(ABC):
    def __init__(self):
        self.is_api = None

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def generate_text(self, prompt, max_length):
        pass

    @abstractmethod
    def save_model(self, save_path):
        pass

    @abstractmethod
    def train_model(self, data_path):
        pass



GPT_3_5_TURBO = "gpt-3.5-turbo-16k"
GPT_4_5_TURBO = "gpt-4-1106-preview"



class ChatGPT(LLMInterface):
    def __init__(self, model_name=GPT_4_5_TURBO):
        super().__init__()

        # Additional initialization code for ChatGPT
        self.is_api = True

        # initialize OpenAI API
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # initialize a clinet for OpenAI API
        self.client = OpenAI()

        # initialize model name
        self.model_name = model_name

        

    def generate_text(self, prompt):

        # prepare request for OpenAI API

        prompt["temperature"] = 0.2

        # send request to OpenAI API
        response = self.client.chat.completions.create(**prompt, model=self.model_name)

        # return response
        return response
    

    def load_model(self, model_path):
        raise NotImplementedError("load_model method is not implemented in ChatGPT class")

    def save_model(self, save_path):
        raise NotImplementedError("save_model method is not implemented in ChatGPT class")

    def train_model(self, data_path):
        raise NotImplementedError("train_model method is not implemented in ChatGPT class")
    


class Llama2(LLMInterface):
    def __init__(self, model_path):
        super().__init__()

        # warn that installed python should 
        # support metal GPU acceleration on Apple Silicon Mac
        print("Warning: Llama2 requires a metal GPU acceleration on Apple Silicon Mac")
    
        # Additional initialization code for Llama
        self.is_api = False

        self.model_path = model_path
        self.prepare()

    
    def prepare(self):        
        # lazy initialize Llama
        self.model = None

        
    def generate_text(self, prompt):

        # if model is not initialized yet, initialize it
        if not self.model:
            self.load_model(self.model_path)

        # prepare request for Llama
        output = self.model(prompt)

        # return response
        return output["choices"][0]["text"]
    

    def load_model(self, model_path):
        try :
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama_cpp import failed, please check your installation")

        self.model = Llama(model_path=self.model_path, n_ctx=4096)    

    def save_model(self, save_path):
        raise NotImplementedError("save_model method is not implemented in Llama class")

    def train_model(self, data_path):
        raise NotImplementedError("train_model method is not implemented in Llama class")
    

