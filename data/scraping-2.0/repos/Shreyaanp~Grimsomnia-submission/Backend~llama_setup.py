import logging
import sys
import requests
import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import torch
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding

#!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install  llama-cpp-python --no-cache-dir
#un comment this to use GPU engine- CUBLAS
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

url = 'https://firebasestorage.googleapis.com/v0/b/ichiropractic.appspot.com/o/test.pdf?alt=media&token=c7b685c1-712d-4b0e-bbfd-3d80198c6584'
if not os.path.exists('Data'):
    os.makedirs('Data')
file_path = os.path.join('Data', 'test.pdf')
response = requests.get(url)
if response.status_code == 200:
    with open(file_path, 'wb') as file:
        file.write(response.content)
else:
    print(f'Failed to download the file: {response.status_code}')


# Setup LlamaCPP
llm = LlamaCPP(
    model_url='',  # compactible model is GGUF only.
    model_path='./dolphin-2.1-mistral-7b.Q4_K_M.gguf', # Here I have use dolphin model from my local machine. please remove this and use your own model path
    temperature=0.1,
    max_new_tokens=3024,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 128},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
print('LlamaCPP is ready to use.')

