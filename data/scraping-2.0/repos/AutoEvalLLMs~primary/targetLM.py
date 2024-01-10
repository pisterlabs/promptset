# Generator.py is a file that contains the generator class and any helper functions necessary:
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import openai
import torch
from collections import OrderedDict, defaultdict
from tqdm import tqdm

DEFAULT_TARGET_GENERATOR_CONFIG = {
    "api_model" : False,
    "max_new_tokens" : 100, # number of tokens to generate outside of the prompt
    "num_generations" : 5, # number of generations per prompt
}

DEFAULT_OPENAI_CONFIG = {
    "model" : "gpt-3.5-turbo",
    "temperature" : 0.7,
    "max_new_tokens" : 100, # num new tokens currently unused
}

DEFAULT_LLAMA2HF_CONFIG = {
    "model" : "meta-llama/Llama-2-7b-hf",
    "tokenizer" : "meta-llama/Llama-2-7b-hf",
    "max_new_tokens": 100,
    # todo decoding strategy
}
# MistralConfig = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

DEFAULT_MISTRAL_CONFIG = {
    "model" : "mistralai/Mistral-7B-Instruct-v0.1",
    "tokenizer" : "mistralai/Mistral-7B-Instruct-v0.1",
    "max_new_tokens": 100,
    # todo decoding strategy
}

DEFAULT_LLAMA_CHAT_CONFIG = {
    "model" : "meta-llama/Llama-2-7b-chat-hf",
    "tokenizer" : "meta-llama/Llama-2-7b-chat-hf",
    "temperature" : 0.7,
    "max_new_tokens" : 100
}


class AbstractConfig(dict):
    """
    Abstract class for config dictionaries
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.update(self.get_defaults())
        self.update(kwargs)

    def get_defaults(self):
        """
        Returns the default config for the class
        """
        raise NotImplementedError
    
    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            return None

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Loads config from yaml file
        """
        return cls(**yaml_path)

class DefaultGenerator():
    """
    Default Class defining the generator for the target language model
    """
    def __init__(self) -> None:
        pass

    def generate(self, input_text, **kwargs):
        """
        Generate text using the language model
        """
        raise NotImplementedError
        
    def generate_from_prompts(self, prompts, num_gen = 1):
        ret = []
        for p in tqdm(prompts):
            ret.append([self.generate(p) for i in range(num_gen)])
        return ret


class OpenAIGenerationConfig(AbstractConfig):
    """
    inheriting from dict allows us to use the [] and .get() methods
    ONLY supports flat config not config of configs
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self["api_key"] = kwargs.get("api_key", os.environ.get("OPENAI_API_KEY"))

    def get_defaults(self):
        return DEFAULT_OPENAI_CONFIG

    
class OpenAIGenerator(DefaultGenerator):
    """
    TODO: add more config parameters
    """
    def __init__(self, config: OpenAIGenerationConfig) -> None:
        super().__init__()
        self.config = config
        self.model = openai.Client(api_key=config.api_key)

    def generate(self, input_text):
        """
        Generates text from openai API
        """
        response = self.model.chat.completions.create(
            messages = [{"role": "user", "content": input_text}], # use user instead of system.
            model = self.config.model,
            temperature = self.config.temperature,
            max_tokens = self.config.max_new_tokens,
        )
        return response.choices[0].message.content
    
class Llama2HFConfig(AbstractConfig):
    """
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def get_defaults(self):
        return DEFAULT_LLAMA2HF_CONFIG
    
class Llama2HFGenerator(DefaultGenerator):
    """
    """
    def __init__(self, config: Llama2HFConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModelForCausalLM.from_pretrained(config.tokenizer, torch_dtype=torch.float16, device_map='cuda')
    
    def generate(self, input_text):
        """
        """
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        # input_ids.to('cuda')
        input_ids = input_ids.to('cuda')
        output = self.model.generate(input_ids, max_length=self.config.max_new_tokens + len(input_ids.flatten()), do_sample=True) # todo make sure this works
        return self.tokenizer.decode(output[0,len(input_ids.flatten()):], skip_special_tokens=True)

class MistralConfig(AbstractConfig):
    """
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def get_defaults(self):
        return DEFAULT_MISTRAL_CONFIG

class MistralGenerator(DefaultGenerator):
    """
    """
    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        self.config = config
        self.model_name = config.model
        self.max_tokens = 100 if config.max_tokens is None else config.max_tokens
        # self.model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=self.config)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map='cuda') # use 16bit floats for vram
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def generate(self, input_text):
        """
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt", return_attention_mask=True)
        input_ids = input_ids.to('cuda')
        token_len = len(input_ids['input_ids'].flatten())
        
        output = self.model.generate(**input_ids, max_length=self.max_tokens + token_len, do_sample=True)
        return self.tokenizer.decode(output[0, token_len:], skip_special_tokens=True)

# class TargetGeneratorConfig(GenerationConfig):
#     """
#     SUPERCLASS for TargetGeneratorConfig
#     api_model: False IF using a weights-based model. Otherwise one of {'gpt', 'claude'}
    
#     This class handles the generation strategy given parameters as follows:

#     greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False
#     contrastive search by calling contrastive_search() if penalty_alpha>0. and top_k>1
#     multinomial sampling by calling sample() if num_beams=1 and do_sample=True
#     beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False
#     beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True
#     diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1
#     constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None
#     assisted decoding by calling assisted_decoding(), if assistant_model is passed to .generate()
#     You do not need to call any of the above methods directly. Pass custom parameter values to ‘.generate()‘. To learn more about decoding strategies refer to the text generation strategies guide.

#     """
#     def __init__(self, api_model: str = False, **kwargs):
#         super().__init__(GenerationConfig, **kwargs)
#         self.api_model = api_model 
#         self.num_generations = kwargs.get('num_generations', 5)
     
