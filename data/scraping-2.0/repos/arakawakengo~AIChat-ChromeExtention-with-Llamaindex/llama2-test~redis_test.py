import os
import sys
import logging
import textwrap

import warnings

warnings.filterwarnings("ignore")


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct"
)
model = AutoModelForCausalLM.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# パイプラインの準備
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

# LLMの準備
llm = HuggingFacePipeline(pipeline=pipe)


from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from typing import Any, List

# 埋め込みクラスにqueryを付加
class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(["query: " + text for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query("query: " + text)

# 埋め込みモデルの準備
embed_model = LangchainEmbedding(
    HuggingFaceQueryEmbeddings(model_name="intfloat/multilingual-e5-large")
)


from llama_index import ServiceContext
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser

# ノードパーサーの準備
text_splitter = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=100,
    paragraph_separator="\n\n",
    tokenizer=tokenizer.encode
)
node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

# サービスコンテキストの準備
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser=node_parser,
)




# stop huggingface warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Uncomment to see debug logs
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import RedisVectorStore
from IPython.display import Markdown, display

loader = SimpleDirectoryReader("/home/Nikkei-intern/intern2023-kyoto-team-basilico/llama2-test/data")
documents = loader.load_data()
for file in loader.input_files:
    print(file)
    # Here is where you would do any preprocessing
    

from llama_index.storage.storage_context import StorageContext

vector_store = RedisVectorStore(
    index_name="test_300_100",
    redis_url="redis://localhost:6379"
)


storage_context = StorageContext.from_defaults(vector_store=vector_store)

import gc

torch.cuda.empty_cache()
gc.collect()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
