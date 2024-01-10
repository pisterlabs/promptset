import os
import streamlit as st
import openai
import re

from dotenv import load_dotenv
from streamlit_chat import message
from langchain.agents import (
    initialize_agent
)
from langchain.chat_models import ChatOpenAI
from langchain.tools.base import (
    BaseTool,
)
from langchain import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    HumanMessage,
    AIMessage
)

from llama_hub.github_repo import GithubClient
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
github_client = GithubClient(os.getenv("GITHUB_TOKEN"))

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_java_info(java_source):
    package_pattern = re.compile(r'package\s+([a-zA-Z_][\w.]*);')
    class_pattern = re.compile(r'(?:public|private|protected|)\s*(?:class|interface|enum)\s+([a-zA-Z_]\w*)')

    # パッケージ名を抽出
    package_match = package_pattern.search(java_source)
    package_name = package_match.group(1) if package_match else None

    # クラス名を抽出
    class_match = class_pattern.search(java_source)
    class_name = class_match.group(1) if class_match else None

    # パッケージ名とクラス名をもとにパスを作成
    if package_name and class_name:
        path = package_name.replace('.', '/') + '/' + class_name + '.java'
        return path
    else:
        return None

llm4 = ChatOpenAI(temperature=0.5, model_name="gpt-4")
llm3_5 = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-16k")

search = GoogleSearchAPIWrapper(google_api_key = google_api_key)
index_directory = "./levica_context"

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except:
    memory = ConversationBufferMemory(return_messages=True)

# チャット履歴（HumanMessageやAIMessageなど）を格納する配列の初期化
history = []

class CustomSearchTool(BaseTool):
    name = "Search"
    description = "useful for when you need to answer questions about current events"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return search.run(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

# 画面部分
st.title("LlamaIndex + LangChain + Local + GPT4 AI for GitHub in Streamlit")
st.caption("by Marthur")

targetDir = st.text_input("対象ディレクトリ")
type = st.text_input("プログラムの種類（ex：.kt）")
local_read_button = st.button("ローカルファイル読み込み")
git_user_input = st.text_input("質問")
git_send_button = st.button("送信")

# ローカルファイル読み込みボタン押下処理
if local_read_button:
    local_read_button = False

    ensure_directory_exists(index_directory)

    llm_predictor = LLMPredictor(llm=llm3_5)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    )

    tools = [
        CustomSearchTool(),
    ]

    docs = SimpleDirectoryReader(
        input_dir = targetDir,
        recursive = True,
        required_exts = type.split(","),
    ).load_data()

    if not os.path.exists(index_directory + "/repository"):
        repository_index = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
        repository_index.storage_context.persist(index_directory + "/repository")
    else:
        storage_context_repo = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_directory + "/repository"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_directory + "/repository"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_directory + "/repository"),
        )
        repository_index = load_index_from_storage(storage_context_repo, service_context=service_context)
    repository_query_engine = repository_index.as_query_engine(service_context=service_context)

    class RepositoryClass(BaseTool):
        name="Repository"
        description = targetDir + "内に存在する" + type + " のプログラム、ソースコードについて情報の取得、表示するために使用します。"

        def _run(self, query: str) -> str:
            """Use the tool."""
            return repository_query_engine.query(query).response
    
        async def _arun(self, query: str) -> str:
            """Use the tool asynchronously."""
            raise NotImplementedError("BingSearchRun does not support async")
    tools.append(RepositoryClass())

    file_list = []
    for doc in docs:
        source = doc.get_text()
        path = extract_java_info(source)
        file_list.append(Document(text = path))

        if not os.path.exists(index_directory + "/source/" + path):
            source_index = GPTVectorStoreIndex.from_documents(documents=[Document(text = source)], service_context=service_context)
            source_index.storage_context.persist(index_directory + "/source/" + path)
        else:
            storage_context_src = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_directory + "/source/" + path),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_directory + "/source/" + path),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_directory + "/source/" + path),
            )
            source_index = load_index_from_storage(storage_context_src, service_context=service_context)
        source_query_engine = source_index.as_query_engine(service_context=service_context)

        class path_class(BaseTool):
            name = path
            description = path + "のソースコードを取得、表示するために使用します。"

            def _run(self, query: str) -> str:
                """Use the tool."""
                return source_query_engine.query(query).response
    
            async def _arun(self, query: str) -> str:
                """Use the tool asynchronously."""
                raise NotImplementedError("BingSearchRun does not support async")
        tools.append(path_class())

    if not os.path.exists(index_directory + "/filelist"):
        file_list_index = GPTVectorStoreIndex.from_documents(documents=file_list, service_context=service_context)
        file_list_index.storage_context.persist(index_directory + "/filelist")
    else:
        storage_context_file_list = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_directory + "/filelist"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_directory + "/filelist"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_directory + "/filelist"),
        )
        file_list_index = load_index_from_storage(storage_context_file_list, service_context=service_context)
    file_list_query_engine = file_list_index.as_query_engine(service_context=service_context)

    class FileListClass(BaseTool):
        name="FileList"
        description="Javaのファイルリスト一覧やファイルパス一覧を表示するために使用します。"

        def _run(self, query: str) -> str:
            """Use the tool."""
            return file_list_query_engine.query(query).response

        async def _arun(self, query: str) -> str:
            """Use the tool asynchronously."""
            raise NotImplementedError("BingSearchRun does not support async")
    tools.append(FileListClass())

    agent = initialize_agent(tools, llm4, agent="zero-shot-react-description", memory=memory, verbose=True)
    agent.save_agent(index_directory + "/agent.json")
    st.session_state["agent"] = agent

    if agent: 
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

if git_send_button :
    git_send_button = False
    agent = st.session_state["agent"]

    memory.chat_memory.add_user_message(git_user_input)

    response = agent.run(git_user_input)
    response = response.replace("mermaid", "")

    memory.chat_memory.add_ai_message(response)
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
