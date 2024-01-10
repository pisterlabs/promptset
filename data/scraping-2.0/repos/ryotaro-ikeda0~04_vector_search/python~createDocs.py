# basic
import traceback
import json
import time
from tqdm import tqdm

# import langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store
from langchain.vectorstores import Qdrant
from langchain.vectorstores.pgvector import PGVector

# dataset
from dataset import (
    WandDataset,
    HomeDepotDataset,
)

# import env
from config import Settings
settings = Settings()

# 全文
def all_embedding():
    try:

        # データセットの読み込み
        WANDS = WandDataset()
        base_docs = WANDS.get_docs()

        # モデルの初期化
        _, embeddings = initialize_model()

        # 原文のまま
        saveVector(embeddings, base_docs, "original")

        # 分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
        docs = text_splitter.split_documents(base_docs)
        saveVector(embeddings, docs, "split")

    except Exception as e:
        traceback.print_exc()


# 1000件のみ
def partial_embedding(n: int = 1000):
    try:
        # jsonファイルの読み込み
        instructions = load_json()
        WANDS = WandDataset(n=n)
        base_docs = WANDS.get_docs()

        # モデルの初期化
        chat, embeddings = initialize_model()

        # 原文のまま
        saveVector(embeddings, base_docs, f"original_{n}")

        # 要約
        for instruction in instructions:
            table_name, instruction = instruction.values()
            docs = analyzeDocuments(chat, base_docs, instruction)
            saveVector(embeddings, docs, table_name)

        # 分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
        docs = text_splitter.split_documents(base_docs)
        saveVector(embeddings, docs, f"split_{n}")

    except Exception as e:
        traceback.print_exc()

# jsonファイルの読み込み
def load_json():
    # 指示文の読み込み
    with open("../data/instructions.json", "r") as f:
        # jsonファイルを読み込んで、辞書型に変換する
        instructions = json.load(f)

    return instructions

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

# ドキュメント解析
def analyzeDocuments(chat, base_docs, instruction):
    qa_chain = load_qa_chain(chat, chain_type="map_reduce", verbose=False)
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

    docs = []
    for doc in tqdm(base_docs):
        try:
            page_content = qa_document_chain.run(input_document=doc.page_content, question=instruction)
        except Exception as e:
            traceback.print_exc()
            page_content = doc.page_content

        docs.append(Document(
            page_content=page_content,
            metadata=doc.metadata
        ))
    return docs

# ベクトルストアに保存
def saveVector(embeddings, docs, table_name):

    # Qdrant
    Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=settings.QDRANT_URL,
        prefer_grpc=True,
        collection_name=table_name,
    )

    # # PGVector
    # CONNECTION_STRING = PGVector.connection_string_from_db_params(
    #     driver="psycopg2",
    #     host="localhost",
    #     port=settings.PORT_PGVECTOR,
    #     database=settings.POSTGRES_DB,
    #     user=settings.POSTGRES_USER,
    #     password=settings.POSTGRES_PASSWORD,
    # )

    # PGVector.from_documents(
    #     embedding=embeddings,
    #     documents=docs,
    #     collection_name=table_name,
    #     connection_string=CONNECTION_STRING,
    # )

if __name__ == "__main__":
    all_embedding()
    partial_embedding(n=1000)
