# llama-index==0.7.0
# Upserting vectors into Pinecone


import json
from llama_index import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
import pinecone
import openai
import os
import logging
import sys
from dotenv import load_dotenv


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if pinecone_api_key is None:
    raise ValueError("Please set your PINECONE_API_KEY environment variable")

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")


folder_list = [
    "./movies/【高校数学1】数と式",
    "./movies/【中1数学】一次方程式",
    "./movies/【中1数学】空間図形",
    "./movies/【中1数学】正の数・負の数",
    "./movies/【中1数学】比例・反比例",
    "./movies/【中1数学】文字式",
    "./movies/【中1数学】平面図形",
    "./movies/【中1数学】資料の活用",
    "./movies/【中2数学】一次関数",
    "./movies/【中2数学】確率",
    "./movies/【中2数学】三角形と四角形",
    "./movies/【中2数学】式の計算",
    "./movies/【中2数学】平行線・多角形・合同",
    "./movies/【中2数学】連立方程式",
    "./movies/【中3数学】三平方の定理",
    "./movies/【中3数学】式の展開と因数分解",
    "./movies/【中3数学】相似な図形",
    "./movies/【中3数学】二次関数",
    "./movies/【中3数学】二次方程式",
    "./movies/【中3数学】平方根",
    "./movies/【中3数学】円",
    "./movies/【高校数学1】集合と命題",
    "./movies/【高校数学1】データの分析/",
    "./movies/【高校数学1】図形と計量",
]


pinecone_index = pinecone.Index("keyword-search")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# define storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)


url_map = {}
with open("URL.json") as f:
    url_map = json.load(f)


def get_url_from_path(path: str):
    value = path.split("/")[-1].split(".")[0]
    urls = [k for k, v in url_map.items() if value == v]
    return urls[0] if len(urls) > 0 else ""


def filename_fn(filename):
    """metadata"""
    return {
        "url": get_url_from_path(filename),
        "file_path": filename,
    }


for folder in folder_list:
    # load documents
    documents = SimpleDirectoryReader(folder, file_metadata=filename_fn).load_data()
    # create index for vectors
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, store_nodes_override=True
    )
