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

# サービスコンテキストの準備
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)


# stop huggingface warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Uncomment to see debug logs
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex
from llama_index.vector_stores import RedisVectorStore
    

vector_store = RedisVectorStore(
    index_name="test_300_100",
    redis_url="redis://localhost:6379"
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

pgQuery = index.as_query_engine()
response = pgQuery.query("習近平ってどんな人？")

print(textwrap.fill(str(response), 100))
