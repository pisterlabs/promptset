import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
import openai
from langchain.schema import BaseOutputParser
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

# .envファイルの読み込み ２階層上の.envを読み込
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv
import openai

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# 環境変数からAPIキーを設定
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

# LLMsの初期化
llm = OpenAI()
chat_model = ChatOpenAI()

# Streamlit UI
st.title("LangSmith Demo")

# ユーザー入力を受け取る
user_input = st.text_input("Enter a prompt:", "What would be a good company name for a company that makes colorful socks?")

# ボタンが押されたら結果を表示
if st.button("Generate Company Name"):
    messages = [HumanMessage(content=user_input)]
    try:
        # OpenAIを使った応答の生成
        llm_response = llm.invoke(user_input)
        chat_model_response = chat_model.invoke(messages)

        # 結果の表示
        st.write("LLM Response:", llm_response)
        st.write("Chat Model Response:", chat_model_response.content if chat_model_response else "No response")

    except Exception as e:
        st.error(f"Error invoking model: {e}")



# >> ['red', 'blue', 'green', 'yellow', 'orange']