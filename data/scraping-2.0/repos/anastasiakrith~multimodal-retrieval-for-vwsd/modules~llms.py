import openai
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM, LlamaTokenizer, LlamaForCausalLM, pipeline

load_dotenv()

class GPT_3:
    """
    Wrapper for openai text-davinci-003
    """

    def __init__(self, max_tokens=70, temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0.6):
        
        openai_api_key = os.getenv("OPENAI_API_KEY")  
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is needed")
        openai.api_key = openai_api_key
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty


    def completion(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            prompt=prompt
        )
        return response.choices[0].text.strip()
    
    
class GPT_3_5:
    """
    Wrapper for openai gpt-3.5-turbo
    """

    def __init__(self, max_tokens=70):
        openai_api_key = os.getenv("OPENAI_API_KEY")  
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is needed")
        openai.api_key = openai_api_key
        
        self.max_tokens = max_tokens

    def completion(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a intelligent assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
    

class BLOOMZ:
    """
    Wrapper for bigscience/bloomz from hugging-face
    """

    def __init__(self, nparameters="1b7", max_tokens=70):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = max_tokens
        if nparameters not in ["1b7", "3b"]:
            raise ValueError("Ivalid nparameters: %s. Should be one of [1b7, 3b]", nparameters)
        
        self.load_from = f"bigscience/bloomz-{nparameters}"
        self.model = AutoModelForCausalLM.from_pretrained(self.load_from, device_map="auto", offload_folder="offload", offload_state_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.load_from)

    def completion(self, prompt):

        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model.generate(**model_input, max_new_tokens=self.max_tokens)
        return self.tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


class OPT:
    """
    Wrapper for bigscience/opt from hugging-face
    """

    def __init__(self, nparameters="2.7b", max_tokens=70):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = max_tokens
        if nparameters not in ["2.7b", "6.7b"]:
            raise ValueError("Ivalid nparameters: %s. Should be one of [2.7b, 6.7b]", nparameters)
        
        self.load_from = f"facebook/opt-{nparameters}"
        self.model = OPTForCausalLM.from_pretrained(self.load_from, device_map="auto", offload_folder="offload", offload_state_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.load_from)

    def completion(self, prompt):

        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model.generate(**model_input, max_new_tokens=self.max_tokens)
        return self.tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


class GALACTICA:
    """
    Wrapper for facebook/galactica-6.7b from hugging-face
    """

    def __init__(self, max_tokens=70):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = max_tokens
        
        self.load_from = "facebook/galactica-6.7b"
        self.model = AutoModelForCausalLM.from_pretrained(self.load_from, device_map="auto", offload_folder="offload", offload_state_dict = True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.load_from)

    def completion(self, prompt):

        model_input = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model.generate(**model_input, max_new_tokens=self.max_tokens)
        return self.tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


class GPT2XL:
    """
    Wrapper for gpt2-xl from hugging-face
    """

    def __init__(self, max_tokens=70):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = max_tokens
        
        self.load_from = "gpt2-xl"
        self.pipeline = pipeline('text-generation', model=self.load_from, device=self.device)

    def completion(self, prompt):

        text = self.generator(prompt, max_length=self.max_tokens)
        return text[0]["generated_text"]