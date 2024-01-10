from .generic import ModelClass
from typing import Callable
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from gpt4all import GPT4All
from langchain.callbacks.manager import CallbackManagerForLLMRun

class RunManager(CallbackManagerForLLMRun):
    def __init__(self, callback):
        self._callback = callback
            
    def on_llm_new_token(self, text: str):
        self._callback(text)
        

class Model(ModelClass):
    def __init__(self):
        self._message_history = []
        # Make sure we have the model we want to use
        print("Downloading / ensuring model exists...", flush=True)
        filename = hf_hub_download(
            repo_id="TheBloke/Vicuna-33B-1-3-SuperHOT-8K-GGML", 
            filename="vicuna-33b-1.3-superhot-8k.ggmlv3.q2_K.bin", 
            cache_dir="/models")
        print("Model download complete", flush=True)
        # Load the model
        filename = try_to_load_from_cache(
            repo_id="TheBloke/Vicuna-33B-1-3-SuperHOT-8K-GGML", 
            filename="vicuna-33b-1.3-superhot-8k.ggmlv3.q2_K.bin", 
            cache_dir="/models")
        print(filename)
        self.model = GPT4All(filename, model_type='gptj', model_path='/models/')
        print("Model loaded", flush=True)
        
    def prompt(self, prompt: str):
        """
        Write a prompt to the bot and return the response.
        """
        return self.model(prompt)
    
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None]) -> None:
        rm = RunManager(callback)
        self.model(prompt, callbacks=[rm])
        print("Prompt all done", flush=True)

    def _callback_wrapper(self, text: str, callback: Callable[[str], None]):
        " Record the response after stripping out the initial prompt "
        callback(self.tokenizer.decode(text, skip_special_tokens=True))