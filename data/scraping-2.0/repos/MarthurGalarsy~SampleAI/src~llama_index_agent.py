import os
import streamlit as st
import openai
import re

from dotenv import load_dotenv
from streamlit_chat import message

from langchain import (
    GoogleSearchAPIWrapper,
    LLMChain
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    HumanMessage,
    AIMessage,
)
from langchain.agents import (
    load_tools,
    ZeroShotAgent,
    AgentExecutor
)
from langchain.chat_models import ChatOpenAI
from langchain.tools.base import (
    BaseTool,
)

from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
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

def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_kotlin_info(kotlin_source):
    package_pattern = re.compile(r'package\s+([a-zA-Z_][\w.]*)')
    class_pattern = re.compile(r'(?:public|private|protected|)\s*(?:class|interface|enum|object)\s+([a-zA-Z_]\w*)')

    # パッケージ名を抽出
    package_match = package_pattern.search(kotlin_source)
    package_name = package_match.group(1) if package_match else None

    # クラス名を抽出
    class_match = class_pattern.search(kotlin_source)
    class_name = class_match.group(1) if class_match else None

    # パッケージ名とクラス名をもとにパスを作成
    if package_name and class_name:
        path = package_name.replace('.', '/') + '/' + class_name + '.kt'
        return path
    else:
        return None

llm4 = ChatOpenAI(temperature=0.5, model_name="gpt-4-1106-preview")
llm3_5 = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-16k")

search = GoogleSearchAPIWrapper(google_api_key = google_api_key)
index_directory = "./index_context"

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

owner = st.text_input("GitHubのOwner")
repository = st.text_input("GitHubのRepository")
type = st.text_input("プログラムの種類（ex：.kt）")
targetDir = st.text_input("対象ディレクトリ")
branch = st.text_input("ブランチ")
git_read_button = st.button("GitHub読み込み")
git_user_input = st.text_input("質問")
git_send_button = st.button("送信")

# GitHub読み込みボタン(LlamaIndex)押下処理
if git_read_button:
    git_read_button = False

    ensure_directory_exists(index_directory)

    llm_predictor = LLMPredictor(llm=llm3_5)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    )

    tools = [
        CustomSearchTool(),
        load_tools(["human"], input_func=get_input)[0]
    ]

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

    if not os.path.exists(index_directory + "/" + repository + "/repository"):
        repository_index = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
        repository_index.storage_context.persist(index_directory + "/" + repository + "/repository")
    else:
        storage_context_repo = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/repository"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/repository"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/repository"),
        )
        repository_index = load_index_from_storage(storage_context_repo, service_context=service_context)
    repository_query_engine = repository_index.as_query_engine(service_context=service_context)

    class RepositoryClass(BaseTool):
        name="Repository"
        description = repository + "内に存在する" + type + " のプログラム、ソースコードについて情報の取得、表示するために使用します。"

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
        path = extract_kotlin_info(source)
        file_list.append(Document(text = path))
        path_class = path + "_class"

        if not os.path.exists(index_directory + "/" + repository + "/source/" + path):
            source_index = GPTVectorStoreIndex.from_documents(documents=[Document(text = source)], service_context=service_context)
            source_index.storage_context.persist(index_directory + "/" + repository + "/source/" + path)
        else:
            storage_context_src = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/source/" + path),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/source/" + path),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/source/" + path),
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

    if not os.path.exists(index_directory + "/" + repository + "/filelist"):
        file_list_index = GPTVectorStoreIndex.from_documents(documents=file_list, service_context=service_context)
        file_list_index.storage_context.persist(index_directory + "/" + repository + "/filelist")
    else:
        storage_context_file_list = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/filelist"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/filelist"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_directory + "/" + repository + "/filelist"),
        )
        file_list_index = load_index_from_storage(storage_context_file_list, service_context=service_context)
    file_list_query_engine = file_list_index.as_query_engine(service_context=service_context)

    class FileListClass(BaseTool):
        name="FileList"
        description="Kotlinのファイルリスト一覧やファイルパス一覧を表示するために使用します。"

        def _run(self, query: str) -> str:
            """Use the tool."""
            return file_list_query_engine.query(query).response

        async def _arun(self, query: str) -> str:
            """Use the tool asynchronously."""
            raise NotImplementedError("BingSearchRun does not support async")
    tools.append(FileListClass())

    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix="""あなたはGitHubやKotlinに詳しいシニアエンジニアです。
        次の質問にできる限り答えてください。次のツールにアクセスできます:""",
#        suffix="""必ずFINAL FANTASY Tacticsのアグリアスの言葉遣いで回答してください。
        suffix="""必ずFINAL FANTASY XIIIのライトニングの言葉遣いで回答してください。
        ただし、分からないことは人間に質問してください。
        質問内容：{question}
        {agent_scratchpad}
        """,
        input_variables=["question", "agent_scratchpad"]
    )
    llm_chain = LLMChain(llm=llm4, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    agent_executor.save_agent(index_directory + "/" + repository + "/agent.json")
    st.session_state["agent_executor"] = agent_executor

    if agent_executor: 
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

if git_send_button :
    git_send_button = False
    agent_executor = st.session_state["agent_executor"]
    memory.chat_memory.add_user_message(git_user_input)

    response = agent_executor.run(git_user_input)
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
