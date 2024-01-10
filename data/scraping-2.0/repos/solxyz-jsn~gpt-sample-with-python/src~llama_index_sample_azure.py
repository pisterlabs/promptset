# -*- coding: utf-8 -*-

"""LlamaIndexサンプル

LlamaIndexを使用して、外部のテキストの情報を参照して
回答をさせるサンプル

"""

import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
    StorageContext,
    load_index_from_storage,
    LangchainEmbedding,
    set_global_service_context,
)
import logging
import sys
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

# 環境変数の読み込み
load_dotenv()

# ログの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)


def create_index(dir_path: str) -> GPTVectorStoreIndex:
    """
    外部データを読み込み、GPTVectorStoreIndexを作成する。

    Parameters
    ----------
    dir_path : str
        外部データのディレクトリのパス。

    Returns
    -------
    GPTVectorStoreIndex
        作成されたGPTVectorStoreIndex。

    """
    # 外部データの読み込み
    documents = SimpleDirectoryReader(os.path.join(dir_path, "data")).load_data()

    # インデックスの作成
    index = GPTVectorStoreIndex.from_documents(documents)

    # インデックスの保存
    index.storage_context.persist()

    return index


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")
# ChatGPT-3.5のモデルのインスタンスの作成
api_key: str = token.token
api_base: str = os.getenv("OPENAI_API_BASE")
api_version: str = os.getenv("OPENAI_API_VERSION")
api_type: str = os.getenv("OPENAI_API_TYPE")
ai_model: str = os.getenv("AZURE_MODEL")
embedding_model: str = os.getenv("AZURE_EMBEDDING_MODEL")

# OpenAIのモデルの作成
llm = AzureChatOpenAI(
    openai_api_type=api_type,
    openai_api_base=api_base,
    openai_api_key=api_key,
    openai_api_version=api_version,
    deployment_name=ai_model,
)

# Embeddingモデルの作成
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        deployment=embedding_model,
        openai_api_key=api_key,
        openai_api_base=api_base,
        openai_api_type=api_type,
        openai_api_version=api_version,
    ),
    embed_batch_size=16,
)

# 実行ファイルのパス
dir_path = os.path.dirname(os.path.realpath(__file__))

# サービスコンテキストの作成
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
)

set_global_service_context(service_context)

# インデックスの作成（初回のみ）
index = create_index(dir_path)

# 保存済みのデータの読み込み（二回目以降）
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

# タイトルの作成
st.title("🐟LlamaIndexのサンプル🐟")
st.text("外部データを読み込んでいます。")

# インプット用のテキストボックスの作成
prompt = st.text_input("質問を入力してください。")

# 入力があったらOpenAIのAPIを実行
if st.button("問い合わせ開始"):
    try:
        response = query_engine.query(
            f"あなたは広報です。可能な限り詳しく日本語で答えてください。また文章に記載のない場合はその旨を教えてください。 ```{prompt}```"
        )
    except Exception as e:
        print("error")
        response = str(e)
    # OpenAIの回答を表示
    st.write(response)
