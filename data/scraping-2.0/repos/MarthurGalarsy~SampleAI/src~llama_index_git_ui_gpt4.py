import os
import streamlit as st
import openai

from dotenv import load_dotenv
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from llama_index import (
    download_loader,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GitLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

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
st.title("LlamaIndex + LangChain + Local + GPT4 AI for GitHub in Streamlit")
st.caption("by Marthur")

place_type = ["Git(LlamaIndex)", "Git(LangChain)","Local(LlamaIndex)"]
place_selector = st.radio("読み込み方切り替え", place_type)
if place_selector == "Git(LlamaIndex)" :
    owner = st.text_input("GitHubのOwner")
    repository = st.text_input("GitHubのRepository")
    type = st.text_input("プログラムの種類（ex：.kt）")
    targetDir = st.text_input("対象ディレクトリ")
    branch = st.text_input("ブランチ")
    git_read_button = st.button("GitHub読み込み")
elif place_selector == "Git(LangChain)" :
    clone_url = st.text_input("GitHubのURL")
    type = st.text_input("プログラムの種類（ex：.kt）")
    branch = st.text_input("ブランチ")
    repo_path = "./temp"
    git_read_button = st.button("GitHub読み込み")
elif place_selector == "Local(LlamaIndex)" :
    targetDir = st.text_input("対象ディレクトリ")
    type = st.text_input("プログラムの種類（ex：.kt）")
    local_read_button = st.button("ローカルファイル読み込み")

target_type = ["Repository", "SingleFile"]
target_type_selector = st.radio("対象切り替え", target_type)
if target_type_selector == "Repository":
    git_user_input = st.text_input("質問")
    git_send_button = st.button("送信")
elif target_type_selector == "SingleFile":
    git_user_input = st.text_input("対象ファイル名")
    gpt_user_input = st.text_input("質問")
    gpt_send_button = st.button("送信")

# GitHub読み込みボタン(LlamaIndex)押下処理
if place_selector == "Git(LlamaIndex)" and git_read_button:
    git_read_button = False

    loader = GithubRepositoryReader(
        github_client,
        owner =                  owner,
        repo =                   repository,
        filter_directories =     ([targetDir], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = (type.split(","), GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
        use_parser =             True,
    )
    docs = loader.load_data(branch=branch)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    )
    index = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
    query_engine = index.as_query_engine(service_context=service_context)
    st.session_state["query_engine"] = query_engine

    if query_engine: 
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

# GitHub読み込みボタン(LangChain)押下処理
if place_selector == "Git(LangChain)" and git_read_button:
    git_read_button = False
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
        embedding=OpenAIEmbeddings(
            disallowed_special=(),
            chunk_size=1
        ), #default
    ).from_loaders([loader])

    st.session_state["index"] = index
    
    if index :
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

# ローカルファイル読み込みボタン押下処理
if place_selector == "Local(LlamaIndex)" and local_read_button:
    local_read_button = False
    docs = SimpleDirectoryReader(
        input_dir = targetDir,
        recursive = True,
        required_exts = type.split(","),
    ).load_data()

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    )
    index = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
    query_engine = index.as_query_engine(service_context=service_context)
    st.session_state["query_engine"] = query_engine

    if query_engine: 
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

# Repositoryで送信ボタン押下処理
if target_type_selector == "Repository" and git_send_button :
    git_send_button = False
    memory.chat_memory.add_user_message(git_user_input)
    if place_selector == "Git(LangChain)":
        index = st.session_state["index"]
        response = index.query(git_user_input)
        response = response.replace("mermaid", "")
    else:
        query_engine = st.session_state["query_engine"]
        response = str(query_engine.query(git_user_input).response)
        response = response.replace("mermaid", "")

    memory.chat_memory.add_ai_message(response)
    st.session_state["memory"] = memory

    # チャット履歴（HumanMessageやAIMessageなど）の読み込み
    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

# SingleFileで送信ボタン押下処理
if target_type_selector == "SingleFile" and gpt_send_button :
    gpt_send_button = False
    git_user_input += "のソースコードを表示してください"
    memory.chat_memory.add_user_message(git_user_input)

    if place_selector == "Git(LangChain)":
        index = st.session_state["index"]
        code_res = index.query(git_user_input)
    else:
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
        model="gpt-4",
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
