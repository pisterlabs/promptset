import os
import streamlit as st
import openai

from dotenv import load_dotenv
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from llama_index import download_loader, GPTVectorStoreIndex
from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
github_client = GithubClient(os.getenv("GITHUB_TOKEN"))

download_loader("GithubRepositoryReader")

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except:
    memory = ConversationBufferMemory(return_messages=True)

# 会話履歴を格納するための変数
conversation_history = []

# チャット履歴（HumanMessageやAIMessageなど）を格納する配列の初期化
history = []

# 画面部分
st.title("langchain for GitHub in Streamlit")
st.caption("by Marthur")

owner = st.text_input("GitHubのOwner")
repository = st.text_input("GitHubのRepository")
type = st.text_input("プログラムの種類（ex：.kt）")
targetDir = st.text_input("対象ディレクトリ")
branch = st.text_input("ブランチ")
read_button = st.button("GitHub読み込み")
model_list = ["Git", "GPT"]
model_selector = st.radio("モデル切り替え", model_list)
if model_selector == "Git":
    git_user_input = st.text_input("質問")
    git_send_button = st.button("送信")
elif model_selector == "GPT":
    git_user_input = st.text_input("対象ファイル名")
    gpt_user_input = st.text_input("質問")
    gpt_send_button = st.button("送信")

# GitHub読み込みボタン押下処理
if read_button:
    read_button = False
    loader = GithubRepositoryReader(
        github_client,
        owner =                  owner,
        repo =                   repository,
        filter_directories =     ([targetDir], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([type], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch=branch)
    index = GPTVectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    st.session_state["query_engine"] = query_engine

    if query_engine: 
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

# Gitで送信ボタン押下処理
if model_selector == "Git" and git_send_button :
    git_send_button = False
    memory.chat_memory.add_user_message(git_user_input)
    query_engine = st.session_state["query_engine"]

    response = query_engine.query(git_user_input).response
    memory.chat_memory.add_ai_message(response)
    st.session_state["memory"] = memory

    # チャット履歴（HumanMessageやAIMessageなど）の読み込み
    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

# Gptで送信ボタン押下処理
if model_selector == "GPT" and gpt_send_button :
    gpt_send_button = False
    git_user_input += "のソースコードを表示してください"
    memory.chat_memory.add_user_message(git_user_input)
    query_engine = st.session_state["query_engine"]

    code_res = query_engine.query(git_user_input).response
    memory.chat_memory.add_ai_message(code_res)

    prompt = "下記のコードがあります。\n下記のコードに対して" + gpt_user_input + "\n" + code_res
    memory.chat_memory.add_user_message(prompt)
    st.session_state["memory"] = memory

    # ユーザーの質問を会話履歴に追加
    conversation_history.append({"role": "user", "content": prompt})
    
    # GPT-4モデルを使用してテキストを生成
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"You are a excellent system engineer."}] + conversation_history,
        max_tokens=3500,
        n=1,
        temperature=0.8,
    )
    gpt_message = gpt_response.choices[0].message['content'].strip()

    # アシスタントの回答を会話履歴に追加
    conversation_history.append({"role": "assistant", "content": gpt_message})
    memory.chat_memory.add_ai_message(gpt_message)
    st.session_state["memory"] = memory

    # チャット履歴（HumanMessageやAIMessageなど）の読み込み
    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

# チャット履歴の表示
for index, chat_message in enumerate(reversed(history)):
    if isinstance(chat_message, HumanMessage):
        message(chat_message.content, is_user=True, key=2 * index)
    elif isinstance(chat_message, AIMessage):
        message(chat_message.content, is_user=False, key=2 * index + 1)
