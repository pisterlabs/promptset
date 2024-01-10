import time
import os
import openai
from pathlib import Path
from typing import List
import streamlit as st
from dotenv import load_dotenv

from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser

import utils
from function_calls import retrieval_func_desc, get_user_information


# Load Environment
load_dotenv('./.env')
pinecone_api_key = os.environ["PINECONE_API_KEY"]
environment = os.environ["PINECONE_ENVIRONMENT"]
openai_api_key = os.environ["OPENAI_API_KEY"]


openai.api_key = openai_api_key
LOG_DIR = "./test_log"
    

@st.cache_resource
def build_query_engine(
    log_dir: str,
):  
    # 会話履歴を読み込んで，nodeを作成
    documents = SimpleDirectoryReader(log_dir).load_data()
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    
    # indexを作成し，Query engineを構築
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine()
    return query_engine


with st.sidebar:
    exit_app = st.sidebar.button("Shut Down")
    if exit_app:  
        # 終了時処理
        utils.save_history(LOG_DIR, st.session_state.messages)
        st.stop()

st.title("ChatBot")

# 初期状態であれば，AI側のメッセージを表示する．
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "こんにちは！"}]
    
# Chat欄に書き込み
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if user_input := st.chat_input():
    
    # Query Engineを構築
    query_engine = build_query_engine(LOG_DIR)
    
    human_msg = {
        "role": "user",
        "content": user_input
    }
    
    # ユーザの入力を保存
    st.session_state.messages.append(human_msg)
    
    # ユーザの入力をChat欄に表示
    st.chat_message("User").write(user_input)
    
    # ChatGPTからの出力を取得
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages,
        functions=retrieval_func_desc,
        function_call="auto"
    ).choices[0].message
    
    if "function_call" in response.keys():
        # Query
        query_response = query_engine.query(user_input).response
        print(f"Context: {query_response}")
        print("---")
        ai_response = {
            "role": "assistant",
            "content": query_response
        }
    else: 
        ai_response = response
    
    # AIからの返答を保存
    st.session_state.messages.append(ai_response)
    
    # AIからの返答をChat欄に表示
    st.chat_message("Assistant").write(ai_response["content"])
    