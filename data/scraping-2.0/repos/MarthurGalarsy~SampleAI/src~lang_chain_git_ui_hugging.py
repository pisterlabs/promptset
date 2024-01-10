import os
import streamlit as st
import openai

from dotenv import load_dotenv
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import GitLoader
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except:
    memory = ConversationBufferMemory(return_messages=True)

st.title("langchain for GitHub in Streamlit")
st.caption("by Marthur")

clone_url = st.text_input("GitHubのURL")
type = st.text_input("プログラムの種類（ex：.kt）")
branch = st.text_input("ブランチ")
repo_path = "./temp"
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

# 会話履歴を格納するための変数
conversation_history = []

# チャット履歴（HumanMessageやAIMessageなど）を格納する配列の初期化
history = []

if read_button:
    read_button = False
    if os.path.exists(repo_path):
        clone_url = None

    loader = GitLoader(
        clone_url=clone_url,
        branch=branch,
        repo_path=repo_path,
        file_filter=lambda file_path: file_path.endswith(type),
    )

    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma, # default
        embedding=HuggingFaceEmbeddings(), #default
    ).from_loaders([loader])

    st.session_state["index"] = index
    
    if index :
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

if model_selector == "Git" and git_send_button :
    git_send_button = False
    memory.chat_memory.add_user_message(git_user_input)
    index = st.session_state["index"]

    response = index.query(git_user_input)
    print(response)

    # セッションへのチャット履歴の保存
    st.session_state["index"] = index
    memory.chat_memory.add_ai_message(response)
    st.session_state["memory"] = memory

    # チャット履歴（HumanMessageやAIMessageなど）の読み込み
    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

if model_selector == "GPT" and gpt_send_button :
    gpt_send_button = False
    memory.chat_memory.add_user_message(git_user_input + "を表示してください")
    index = st.session_state["index"]

    code_res = index.query(git_user_input + "を表示してください")

    # セッションへのチャット履歴の保存
    st.session_state["index"] = index
    memory.chat_memory.add_ai_message(code_res)
    st.session_state["memory"] = memory

    prompt = "下記のコードがあります。\n下記のコードに対して" + gpt_user_input + "\n" + code_res
    memory.chat_memory.add_user_message(prompt)

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
