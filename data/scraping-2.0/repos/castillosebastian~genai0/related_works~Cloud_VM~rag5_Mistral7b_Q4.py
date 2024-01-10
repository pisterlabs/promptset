"""

!pip install -q pypdf
!pip install -q python-dotenv

!pip install -q transformers

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install  llama-cpp-python --no-cache-dir

!pip install -q llama-index

!pip -q install sentence-transformers

"""

import logging
import sys
import requests
import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import torch
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding
import sentence_transformers

# Start script
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

dirpath = 'related_works/Cloud_VM/'
filename = dirpath + 'ey.pdf'
url = 'https://assets.ey.com/content/dam/ey-sites/ey-com/nl_nl/topics/jaarverslag/downloads-pdfs/2022-2023/ey-nl-financial-statements-2023-en.pdf'

if not os.path.exists(filename):
    print(f"Downloading {filename} from {url}...")    
    response = requests.get(url)
    with open(dirpath + 'ey.pdf', 'wb') as f:
        f.write(response.content)

documents = SimpleDirectoryReader(
    input_files=[filename]
).load_data()

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("Explain to me the Statement of profit or loss of Ernst & Young Nederland") # 30G RAM y full 24 vCPU
print(response)

# conversation-like
while True:
  query=input()
  response = query_engine.query(query)
  print(response)

