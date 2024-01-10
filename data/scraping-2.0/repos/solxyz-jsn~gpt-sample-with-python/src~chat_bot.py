# -*- coding: utf-8 -*-

"""ChatGPTクローン

OpenAI APIを使用して、ChatGPTを再現したサンプル

"""

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
import logging
import sys
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ログの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# ChatGPT-3.5のモデルのインスタンスの作成
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except:
    memory = ConversationBufferMemory(return_messages=True)

# チャット用のチェーンのインスタンスの作成
chain = ConversationChain(llm=llm, memory=memory, verbose=True)

# タイトルの作成
st.title("🐟チャットの単純な実装🐟")

# チャット履歴を表示するためのコンテナ
c = st.container()

# 入力フォームの作成
prompt = st.text_input("メッセージを入力してください。")

# チャット履歴配列の初期化
history = [
    AIMessage(content="こんにちは。お手伝いできることはありますか？", additional_kwargs={}, example=False)
]

# 質問が入力された時、OpenAIのAPIを実行
if prompt:
    # ChatGPTの実行
    chain(prompt)
    # チャット履歴の取得
    st.session_state["memory"] = memory
    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

# チャット履歴の表示
for index, chat_message in enumerate(history):
    # ユーザーのメッセージの場合
    if type(chat_message) == HumanMessage:
        with c.chat_message("user", avatar="🧑"):
            st.write(chat_message.content)
    # AIのメッセージの場合
    elif type(chat_message) == AIMessage:
        with c.chat_message("agent", avatar="🤖"):
            st.write(chat_message.content)
