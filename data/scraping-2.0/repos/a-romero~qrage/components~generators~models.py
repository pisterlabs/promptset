import os
from typing import Dict, Any
from llama_index.llms import Anthropic, HuggingFaceLLM, OpenAI

class Model:

    def __init__(
            self,
            generative_model: str,
            context_window: int = 512,
            temperature: int = 0,
            max_length: int = 4096,
        ):
        """
        :param generative_model: The generative model to use. Available options are 'mistral', 'falcon', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', and 'command'.
        :param context_window: The context window to use for the generative model. Default: 512.
        :param temperature: The temperature to use for the generative model. Default: 0.0.
        :param max_length: The maximum length of the generated text. Default: 4096.
        """
        
        self.context_window = context_window
        self.temperature = temperature
        self.max_length = max_length

        models_config: Dict[str, Any] = {
                "mistral": {"model_name": "mistralai/Mistral-7B-Instruct-v0.1",
                            "api_key": os.getenv('HUGGINGFACEHUB_API_TOKEN')},
                "falcon": {"model_name": "tiiuae/falcon-7b-instruct",
                            "api_key": os.getenv('HUGGINGFACEHUB_API_TOKEN')},
                "gpt-3.5-turbo": {"model_name": "gpt-3.5-turbo",
                            "api_key": os.getenv('OPENAI_API_KEY')},
                "gpt-4": {"model_name": "gpt-4",
                            "api_key": os.getenv('OPENAI_API_KEY')},
                "gpt-4-turbo": {"model_name": "gpt-4-1106-preview",
                            "api_key": os.getenv('OPENAI_API_KEY')},
                "command": {"model_name": "command",
                            "api_key": os.getenv('COHERE_API_KEY')},
            }

        model: Dict[str, str] = models_config[generative_model]
        self.model_name = model["model_name"]
        self.api_key = model["api_key"]
    
    def __str__(self):
        return f"Model: {self.model_name}, API Key: {self.api_key}, Context Window: {self.context_window}, Temperature: {self.temperature}, Max Length: {self.max_length}"
    
    def __repr__(self):
        return f"Model: {self.model_name}, API Key: {self.api_key}, Context Window: {self.context_window}, Temperature: {self.temperature}, Max Length: {self.max_length}"


    def baseModel(self, model_name: str):
        return getattr(self, model_name.replace("-", "_"))()

    def mistral(self):
        return HuggingFaceLLM(
            model=self.model_name,
            context_window=self.context_window,
            generate_kwargs={"temperature":self.temperature},
            tokenizer_kwargs={"max_length":self.max_length},
            )
    
    def falcon(self):
        return HuggingFaceLLM(
            model=self.model_name,
            context_window=self.context_window,
            generate_kwargs={"temperature":self.temperature},
            tokenizer_kwargs={"max_length":self.max_length},
            )
    
    def gpt_3_5_turbo(self):
        return OpenAI(
            model=self.model_name,
            temperature=self.temperature,
            )

    def gpt_4(self):
        return OpenAI(
            model=self.model_name,
            temperature=self.temperature,
            )
    
    def gpt_4_turbo(self):
        return OpenAI(
            model=self.model_name,
            temperature=self.temperature,
            )
    
    #def command(self):
    #    return Anthropic(
    #        model=self.model_name,
    #        )
