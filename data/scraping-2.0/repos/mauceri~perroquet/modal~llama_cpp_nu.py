import json
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import re

from llama_cpp import Llama

import modal
import logging
import subprocess

from pydantic import BaseModel

MODEL_DIR = "/root/models"

def download_model_to_folder():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading model")
    # subprocess.run(['huggingface-cli',
    #                 'download',
    #                 'TheBloke/Vigogne-2-7B-Chat-GGUF',
    #                 'vigogne-2-7b-chat.Q5_K_M.gguf', 
    #                 '--local-dir',
    #                 MODEL_DIR,
    #                 '--local-dir-use-symlinks',
    #                 'False'])
    subprocess.run(['huggingface-cli',
                    'download',
                    'TheBloke/Vigostral-7B-Chat-GGUF',
                    'vigostral-7b-chat.Q5_K_M.gguf', 
                    '--local-dir',
                    MODEL_DIR,
                    '--local-dir-use-symlinks',
                    'False'])
  
  

    logging.info(os.listdir("/root/models/"))



image = (
    modal.Image.debian_slim()
    .run_commands("apt-get update")
    .run_commands("apt-get install -y build-essential cmake libopenblas-dev pkg-config")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch")
    .pip_install("huggingface")
    .pip_install("huggingface_hub")
    .run_commands("pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")
    .run_function(download_model_to_folder,secret=modal.Secret.from_name("llama-cpp-hf"))
    .pip_install("langchain")
)


stub = modal.Stub("llama-cpp-python-nu", image=image)

@stub.function()
def install_llama_cpp() :
    import torch
    from llama_cpp import Llama
    from langchain.llms import LlamaCpp
    """
    from langchain.prompts import ChatPromptTemplate
    from langchain import PromptTemplate, LLMChain
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    """

    import subprocess
    import time
    start = time.time()
    tstart = start

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    prompt = """
    <|system|>: Vos réponses sont concises
    <|user|>: {q}
    """
    # TheBloke/Vigogne-2-7B-Chat-GGUF

    #llm = LlamaCpp(model_path="/root/models/vigogne-2-7b-chat.Q5_K_M.gguf")
    #llm = Llama(model_path="/root/models/vigogne-2-7b-chat.Q5_K_M.gguf")
    #llm = Llama(model_path="/root/models/vigostral-7b-chat.Q5_K_M.gguf")
    

@stub.local_entrypoint()
def main():
    install_llama_cpp.remote()
    

ex = """
<s>[INST] <<SYS>>
Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. 
Aidez autant que vous le pouvez.
<</SYS>>
Bonjour ! Comment ça va aujourd'hui ? [/INST] 
Bonjour ! Je suis une IA, donc je n'ai pas de sentiments, mais je suis prêt à vous aider. Comment puis-je vous assister aujourd'hui ? </s>
[INST] Quelle est la hauteur de la Tour Eiffel ? [/INST] 
La Tour Eiffel mesure environ 330 mètres de hauteur. </s>
[INST] Comment monter en haut ? [/INST]"""
@stub.cls()
class Model:
    def __enter__(self):
        self.prompt = """
             <s>[system]<<SYS>> Vos réponses sont concises<<SYS>>
             {c}
             [INST]{q}[/INST]
             """
        #self.prompt = """
        #    <s>[system]<<SYS>> Vos réponses sont concises<<SYS>>
        #    [INST]{q}[/INST]
        #    """

        self.instructions = [{"role":"system","content":"Vos réponses sont concises"}]

        self.llm =Llama(model_path="/root/models/vigostral-7b-chat.Q5_K_M.gguf",n_ctx=4096)
        
        # Load the model. Tip: MPT models may require `trust_remote_code=true`.

    @modal.method()
    def generate(self, question):
            
        for i in range(10):
            print(f"Essai n° {i}")
            self.llm.reset()
            prompt_tokens: List[int] = (
                self.llm.tokenize(question.encode("utf-8"))
                    if question != "" else [self.token_bos()]
            )
            print(f"Voici le prompt: \"{question}\" sa longueur est de {len(prompt_tokens)} tokens")
            result = self.llm.create_completion(
                prompt=question,max_tokens=1024,
            )
            print(f"retour{result}")
            #retour{'id': 'cmpl-7aae2a66-7da8-4e65-b5a9-c6d859a179b8', 
            #       'object': 'text_completion', 
            #       'created': 1701878289, 
            #       'model': '/root/models/vigostral-7b-chat.Q5_K_M.gguf', 
            #       'choices': [{'text': ' Henri IV est un roi de France. Il a régné du 1', 
            #                        'index': 0, 'logprobs': None, 'finish_reason': 'length'}], 
            #        'usage': {'prompt_tokens': 218, 'completion_tokens': 16, 'total_tokens': 234}}
            
           
            if result["choices"][0]["text"].strip() != "":
                print(f"Voici le résultat {result}")
                return result
           
        raise Exception("No reply")
 
gmodel = Model()


web_app = FastAPI()

class Item(BaseModel):
    question: str



@web_app.get("/")
async def handle_root(user_agent: str = Header(None)):
    logging.info(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/question")
async def handle_question(item:Item):
    answer = None  
    try:
        print(f"****************************************** Question {item.question}")
        # Logique si context n'est pas fourni
        answer = gmodel.generate.remote(item.question)
    except:
        raise Exception("Quelque chose n'a pas fonctionné") 
    print(f"****************************************** Réponse : {answer}")
    return answer



@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.deploy("webapp")
