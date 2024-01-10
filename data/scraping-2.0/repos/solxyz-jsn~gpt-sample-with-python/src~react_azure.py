# -*- coding: utf-8 -*-

"""ReActサンプル

LangChainを使用して、ReActを実行するサンプル

"""
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
import logging
import sys
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

# 環境変数の読み込み
load_dotenv()

# ログの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# Azure ADの認証
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

# 各種環境変数の読み込み
api_key: str = token.token
api_base: str = os.getenv("OPENAI_API_BASE")
api_version: str = os.getenv("OPENAI_API_VERSION")
api_type: str = os.getenv("OPENAI_API_TYPE")
ai_model: str = os.getenv("AZURE_MODEL")
embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL")

# OpenAIのインスタンスの作成
llm = AzureChatOpenAI(
    openai_api_type=api_type,
    openai_api_base=api_base,
    openai_api_key=api_key,
    openai_api_version=api_version,
    deployment_name=ai_model,
)

# 検索ツールの読み込み
tools = load_tools(["serpapi"], llm=llm)

# エージェントの初期化
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# タイトルの作成
st.title("🐟ReActのサンプル🐟")

# インプット用のテキストボックスの作成
prompt = st.text_input("質問を入力してください。")

# 入力があったらOpenAIのAPIを実行
if st.button("問い合わせ開始"):
    try:
        response = agent.run(
            f"質問について、丁寧な日本語で答えてください。```質問：{prompt}```",
        )
    except Exception as e:
        print("error")
        response = str(e)
    # OpenAIの回答を表示
    st.write(response)
