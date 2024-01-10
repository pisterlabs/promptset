from llama_cpp import Llama
import os
from openai import OpenAI

class LanguageModelClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def create_chat_completion(self, messages: list, max_tokens: int):
        raise NotImplementedError
    

class OpenAIClient(LanguageModelClient):
    def __init__(self, api_key: str):
        super().__init__(model_name="gpt-4-1106-preview")
        self.client = OpenAI(api_key=api_key)

    def create_chat_completion(self, messages: list, max_tokens: int):
        return self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    
class HermesClient(LanguageModelClient):
    def __init__(self, model_path: str, n_ctx: int, n_batch: int, chat_format: str, verbose: bool):
        super().__init__(model_name="openhermes-2.5-mistral-7b")
        hermes_params = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "chat_format": chat_format,
            "verbose": verbose
        }
        
        operating_system = os.getenv("OPERATING_SYSTEM")
        if operating_system == "Windows":
            hermes_params["n_gpu_layers"] = 50

        self.client = Llama(**hermes_params)

    def create_chat_completion(self, messages: list, max_tokens: int):
        response = self.client.create_chat_completion(messages=messages, max_tokens=3, temperature=0.0)
        return response
    
class LlamaClient(LanguageModelClient):
    def __init__(self, model_path: str, n_ctx: int, n_batch: int, chat_format: str, verbose: bool):
        super().__init__(model_name="llama-2-7B")
        llama_params = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "chat_format": chat_format,
            "verbose": verbose
        }
        
        operating_system = os.getenv("OPERATING_SYSTEM")
        if operating_system == "Windows":
            llama_params["n_gpu_layers"] = 50

        self.client = Llama(**llama_params)

    def create_chat_completion(self, messages: list, max_tokens: int):
        response = self.client.create_chat_completion(messages=messages, temperature=0.0, max_tokens=2)
        
        return response