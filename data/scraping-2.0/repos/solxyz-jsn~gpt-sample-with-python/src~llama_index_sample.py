# -*- coding: utf-8 -*-

"""LlamaIndexサンプル

LlamaIndexを使用して、外部のテキストの情報を参照して
回答をさせるサンプル

"""

import streamlit as st
from langchain.chat_models import ChatOpenAI
from llama_index import (
    QuestionAnswerPrompt,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
    StorageContext,
    load_index_from_storage,
)
import openai
import logging
import sys
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# Open AI APIキーの設定
openai.api_key = os.environ["OPENAI_API_KEY"]

# ログの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# サービスコンテキストの作成
service_context = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))
)

# # 実行ファイルのパス
# dir_path = os.path.dirname(os.path.realpath(__file__))

# # 外部データの読み込み（＊初回のみ実行）
# documents = SimpleDirectoryReader(os.path.join(dir_path, "data")).load_data()
# # インデックスの作成（＊初回のみ実行）
# index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
# # インデックスの保存（＊初回のみ実行）
# index.storage_context.persist()

# 保存済みのデータの読み込み
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# タイトルの作成
st.title("🐟LlamaIndexのサンプル🐟")
st.text("外部データを読み込んでいます。")

# インプット用のテキストボックスの作成
prompt = st.text_input("質問を入力してください。")

# 入力があったらOpenAIのAPIを実行
if prompt:
    try:
        response = query_engine.query(
            f"あなたは広報です。簡潔な日本語で答えてください。また文章に記載のない場合はその旨を教えてください。 ```{prompt}```"
        )
    except Exception as e:
        print("error")
        response = str(e)
    # OpenAIの回答を表示
    st.write(response.response)
