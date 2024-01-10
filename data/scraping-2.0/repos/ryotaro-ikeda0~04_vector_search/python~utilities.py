# basic
import traceback
import json
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

# import langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.docstore.document import Document

# Vector Store
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.vectorstores.pgvector import PGVector

# import env
from config import Settings
from langchain import PromptTemplate
settings = Settings()

# ドキュメントの読み込み
def load_docs():
    # テキストデータの読み込み
    with open("../data/baseDocs.json", "r") as f:
        # jsonファイルを読み込んで、辞書型に変換する
        data = json.load(f)
        base_docs = [Document(page_content=text["page_content"], metadata=text["metadata"]) for text in data]

    return base_docs

# モデルの初期化
def initialize_model():
    chat = AzureChatOpenAI(
        temperature=0,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        openai_api_base=settings.AZURE_OPENAI_API_ENDPOINT,
        deployment_name=settings.AZURE_OPENAI_API_DEPLOYMENT_NAME,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        openai_api_type="azure",
        max_tokens=1000,
    )

    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        openai_api_base=settings.AZURE_OPENAI_API_ENDPOINT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        openai_api_type="azure",
        model=settings.EMBEDDING_MODEL_NAME,
        chunk_size=1
    )
    return chat, embeddings

# 3つ以上のDataFrameをマージする
def pd_multi_merge(dfs: list, columns, **kwargs) -> pd.DataFrame:
    for i, column in enumerate(columns):
        dfs[i].columns = [column]

    df = dfs[0]
    for i in range(1, len(dfs)):
        df = pd.merge(df, dfs[i], **kwargs)
    return df