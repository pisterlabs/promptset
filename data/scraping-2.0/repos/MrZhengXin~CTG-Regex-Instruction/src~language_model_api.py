from abc import ABCMeta, abstractmethod
from huggingface_hub import InferenceApi
import openai
import asyncio
# from EdgeGPT import Chatbot
import json
import requests

class LanguageModelAPI:
    __metaclass__ = ABCMeta
    @abstractmethod
    def infer(self):
        pass

class BLOOM(LanguageModelAPI):
    def __init__(self, model, token="") -> None:
        self.inference = InferenceApi("bigscience/"+model,token=token)
        self.API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
        self.headers = {"Authorization": "Bearer xxx"}

    def infer(self, 
            prompt,
            max_length = 128,
            top_k = 0,
            num_beams = 0,
            no_repeat_ngram_size = 2,
            top_p = 0.9,
            seed=42,
            temperature=0.7,
            greedy_decoding = False,
            return_full_text = False,
            stop=None):
        top_k = None if top_k == 0 else top_k
        do_sample = True # False if num_beams > 0 else not greedy_decoding
        num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
        no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
        top_p = None if num_beams else top_p
        early_stopping = None if num_beams is None else num_beams > 0

        params = {
            # "inputs": prompt,
            # "max_new_tokens": max_length,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "do_sample": do_sample,
            "seed": seed,
            "early_stopping":early_stopping,
            "no_repeat_ngram_size":no_repeat_ngram_size,
            "num_beams":num_beams,
            "return_full_text":return_full_text,
            "use_cache": False
        }
        response = self.inference(prompt, params=params)
        # response = requests.post(self.API_URL, headers=self.headers, json=payload).json()
        len_prompt = len(prompt)
        output_text = response[0]['generated_text'][len_prompt:].strip()
        return output_text

class GPT3(LanguageModelAPI):
    def __init__(self, model="text-davinci-003", token="") -> None:
        openai.api_key = token
        self.model = model

    def infer(self, prompt, temperature=0.7, max_tokens=256, stop='\n'):
        response = openai.Completion.create(engine=self.model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, stop=stop, request_timeout=10)
        # openai.Completion.create(engine="text-davinci-003", prompt="hello, world", request_timeout=10)
        try:
            output_text = response["choices"][0]["text"]
        except:
            output_text = ""
        return output_text

class ChatGPT(LanguageModelAPI):
    def __init__(self, model="gpt-3.5-turbo", token="") -> None:
        openai.api_key = token
        self.model = model

    def infer(self, prompt, temperature=1, max_tokens=1024, stop=None, top_p=1.0):
        response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens, stop=stop, top_p=1.0)
        # openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "prompt"}], )
        output_text = response["choices"][0]["message"]["content"]
        return output_text
        
    def infer_batch(self, prompt_list, temperature=1, max_tokens=1024, stop=None, top_p=1.0):
        response = openai.ChatCompletion.create(model=self.model, messages=[[{"role": "user", "content": prompt}] for prompt in prompt_list], temperature=temperature, max_tokens=max_tokens, stop=stop, top_p=1.0)
        output_text = [r["choices"][0]["message"]["content"] for r in response]
        return output_text
