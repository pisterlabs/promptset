import configparser
import os
from pathlib import Path
import urllib.request

import numpy as np
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, GPTQConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.llms import LlamaCpp

from RAG.output_formatter import strip_all

def get_model_cfg():
    config = configparser.ConfigParser()
    config.read(os.path.join(Path(__file__).absolute().parent, "model_config.cfg"))
    return config

def choose_bot(model_name=None, model_params=None, gen_params=None, q_bits=None):
    if model_name is None:
        model_cfg = get_model_cfg()
        models = model_cfg.sections()
        model_families = dict({str(k): v for k, v in enumerate(sorted(set([model.split("-")[0] for model in models])))})
        print("Here are the available model families, please choose one:\n")
        for i, repo in model_families.items():
            print(f"{i}: {repo}")  
        while True:
            model_family_id = input()
            model_family = model_families.get(model_family_id)
            if model_family is None:
                print("Please select from one of the options!")
            else:
                break
        num_repo = dict({str(k): v for k, v in enumerate([model for model in models if model_family in model])})
        print("\nChoose a version:\n")
        for i, repo in num_repo.items():
            repo_name = repo.replace("_", "-")
            print(f"{i}: {repo_name}")  
        while True:
            model_id = input()
            model_name = num_repo.get(model_id)
            if model_name is None:
                print("Please select from one of the options!")
            else:
                break
    model_dict = {
        "VICUNA": Vicuna,
        "LLAMA2": LLaMA2,
        "BELUGA": StableBeluga,
        "CHATGPT": ChatGPT,
        "CLAUDE": Claude,
        "MISTRAL": Mistral,
        "WIZARDLM": WizardLM,
        "ZEPHYR": Zephyr,
        "OPENCHAT": OpenChat,
        "STARLING": Starling,
        "YI": Yi,
        "SOLAR": Solar
    }
    model = model_name.split("-")[0]
    if model in ["CHATGPT", "CLAUDE"]:
        return model_dict[model](model_name, model_params, gen_params)
    else:
        return model_dict[model](model_name, model_params, gen_params, q_bits)

class Chatbot:

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        self.cfg = get_model_cfg()[model_name]
        self.name = model_name
        self.repo_id = self.cfg.get("repo_id")
        self.model_basename = self.cfg.get("basename")
        self.context_length = self.cfg.get("context_length")
        self.q_bit = q_bits
        self.model_type = self.get_model_type()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params(model_params)
        self.gen_params = self.get_gen_params(gen_params)
        self.model = self.init_model()
        self.pipe = self.init_pipe()

    def prompt_template(self):
        return None
    
    def prompt_chatbot(self, prompt):
        return strip_all(self.prompt_template()).format(prompt=strip_all(prompt))
    
    def count_tokens(self, prompt):
        if isinstance(prompt, str):
            return len(self.tokenizer(prompt).input_ids)
        if isinstance(prompt, list):
            return [len(self.tokenizer(chunk).input_ids) for chunk in prompt]
        
    def find_best_k(self, chunks, prompt, strategy="optim"):
        avg_chunk_len = np.mean(self.count_tokens(chunks))
        avail_space = int(self.context_length) - self.count_tokens(prompt)
        if strategy == "max":
            pass
        elif strategy == "optim":
            avail_space /= 2
        return int(np.floor(avail_space/avg_chunk_len))

    def get_model_type(self):
        if self.repo_id.endswith("GPTQ"):
            return "GPTQ"
        elif self.repo_id.endswith("GGUF") or self.repo_id.endswith("GGML"):
            return "GGUF"
        elif self.repo_id.endswith("AWQ"):
            return "AWQ"
        elif self.name in ["CLAUDE", "GPT"]:
            return "proprietary"
        else:
            return "default"
        
    def init_tokenizer(self):
        if self.model_type == "GGUF":
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        elif self.model_type == "proprietary":
            return None
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self, gen_params):
        if self.model_type == "GGUF" or self.name == "GPT":
            name_token_var = "max_tokens"
        elif self.name == "CLAUDE":
            name_token_var = "max_tokens_to_sample"
        else:
            name_token_var = "max_new_tokens"
        if gen_params is None:
            return {
            name_token_var: 512,
            #"temperature": 0.7,
            }
        elif "max_new_tokens" or "max_tokens_to_sample" in gen_params.keys():
            value = gen_params.pop("max_new_tokens")
            gen_params[name_token_var] = value
        return gen_params
    
    def ggum_params(self):
        rope_freq_scale = float(self.cfg.get("rope_freq_scale")) if self.cfg.get("rope_freq_scale") else 1
        return {
                "n_gpu_layers": -1,
                "n_batch": 512,
                "verbose": False,
                "n_ctx": self.context_length,
                "rope_freq_scale": rope_freq_scale
                }
    
    def gptq_params(self):
        config = GPTQConfig(max_input_length=self.context_length)
        return {"quantization_config": config,
                "revision": "main"}
    
    def default_model_params(self):
        return {}
    
    def get_model_params(self, model_params):
        if model_params is None:
            if self.model_type == "GGUF":
                return self.ggum_params()
            else:
                return self.default_model_params()
        else:
            return model_params
    
    def init_model(self):
        if self.name == "CLAUDE":
            return ChatAnthropic(model=self.repo_id, **self.gen_params)
        elif self.name == "GPT":
            return ChatOpenAI(model=self.repo_id, **self.gen_params)
        elif self.model_type == "GGUF":
            if os.getenv("HF_HOME") is None:
                hf_cache_path = os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "transformers")
            else:
                hf_cache_path = os.getenv("HF_HOME")
            model_folder = os.path.join(hf_cache_path, self.repo_id.replace("/", "-"))
            bit_range = range(2, 9)
            if self.q_bit not in bit_range:
                print("This is a quantized model, please choose the number of quantization bits: ")
                for i in bit_range:
                    print(f"{i}")  
                while True:
                    q_bit = input()
                    if q_bit.isdigit():
                        if int(q_bit) not in bit_range:
                            print("Please select from one of the options!")
                        else:
                            self.q_bit = q_bit
                            break
                    else:
                        print("Please enter a number!")
            self.model_basename = "-".join(self.repo_id.split('/')[1].split("-")[:-1]).lower()
            if self.q_bit in [2, 6]:
                self.model_basename = f"{self.model_basename}.Q{self.q_bit}_K.gguf"
            elif self.q_bit == 8:
                self.model_basename = f"{self.model_basename}.Q{self.q_bit}_0.gguf"
            else:    
                self.model_basename = f"{self.model_basename}.Q{self.q_bit}_K_M.gguf"
            model_url_path = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.model_basename}"
            if not os.path.exists(os.path.join(model_folder, self.model_basename)):
                os.makedirs(model_folder, exist_ok=True)
                try:
                    print("Downloading model!")
                    urllib.request.urlretrieve(model_url_path, os.path.join(model_folder, self.model_basename))
                except Exception as e:
                    print(e)
                    print("Couldn't find the model, please choose again! (Maybe the model isn't quantized with this bit?)")
            return LlamaCpp(
                    model_path=os.path.join(model_folder, self.model_basename),
                    **self.model_params,
                    **self.gen_params)     
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    low_cpu_mem_usage=True,
                    device_map="auto")
        
    def init_pipe(self):            
        if self.model_type in ["GGUF", "proprietary"]:
            return self.model
        else:
            return HuggingFacePipeline(pipeline=pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, **self.gen_params))

class Vicuna(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user"s questions.
        USER: 
        {prompt}
        ASSISTANT:"""
    
class LLaMA2(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.<</SYS>>
                {prompt}
                [/INST]"""
    
    def default_model_params(self):
        return {
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "token": True,
                }
    
class StableBeluga(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """### System: 
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.
        ### User: 
        {prompt}
        ### Assistant:"""     

class Claude(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None) -> None:
        super().__init__(model_name, model_params, gen_params)

    def prompt_template(self):
        return """Human: {prompt}
        Assistant:"""
    
    def count_tokens(self, prompt):
        if isinstance(prompt, str):
            return self.model.count_tokens(prompt)
        if isinstance(prompt, list):
            return max([self.model.count_tokens(chunk) for chunk in prompt])
        
class ChatGPT(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None) -> None:
        super().__init__(model_name, model_params, gen_params)

    def prompt_template(self):
        return """{prompt}"""
    
    def count_tokens(self, prompt):
        if isinstance(prompt, str):
            return self.model.get_num_tokens(prompt)
        if isinstance(prompt, list):
            return max([self.model.get_num_tokens(chunk) for chunk in prompt])
        
class Mistral(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """<s>[INST] {prompt} [/INST]"""

class Zephyr(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """<|system|></s>
                         <|user|>
                         {prompt}</s> 
                         <|assistant|>"""

class WizardLM(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user"s questions.
        USER: 
        {prompt}
        ASSISTANT:""" 
    
class OpenChat(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:"""
    
class Starling(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"""
    
class Yi(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """<|im_start|>system<|im_end|>
        <|im_start|>user{prompt}<|im_end|>
        <|im_start|>assistant"""
    
class Solar(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return """### User:
        {prompt}
        ### Assistant:
        """