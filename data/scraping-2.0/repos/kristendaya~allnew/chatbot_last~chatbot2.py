# 필요한 라이브러리 임포트

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import os
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import weaviate
from langchain.vectorstores.weaviate import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = ""
os.environ["WEAVIATE_API_KEY"] = ""

# 문서 로더 설정
doc_loader = DirectoryLoader(
    'data/',
    glob='./*.pdf',
    show_progress=True
)
docs = doc_loader.load()

# 텍스트 스플리터 설정
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)
splitted_docs_list = splitter.split_documents(docs)

# Weaviate 클라이언트 및 인증 설정
auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get('WEAVIATE_API_KEY'))
client = weaviate.Client(
    url="https://langchain-wa8isf85.weaviate.network",
    auth_client_secret=auth_config,
    additional_headers={
        "X-OpenAI-Api-Key": os.environ.get('OPENAI_API_KEY')
    }
)

# Weaviate 클래스 스키마 설정
class_obj = {
    "class": "LangChain",
    "vectorizer": "text2vec-openai",
}

# 이미 생성된 클래스가 아닌 경우 스키마에 클래스를 추가
try:
    client.schema.create_class(class_obj)
except:
    print("Class already exists")

# 임베딩 모델 설정 (OpenAI 임베딩 사용)
embeddings = OpenAIEmbeddings()

# 벡터스토어 설정 (Weaviate를 사용한 벡터스토어)
vectorstore = Weaviate(client, "LangChain", "text", embedding=embeddings)
documents = splitted_docs_list

# 문서 텍스트 및 메타데이터 추출
texts = [d.page_content for d in documents]
metadatas = [d.metadata for d in documents]

# 벡터스토어에 텍스트 추가 (임베딩 및 메타데이터 포함)
vectorstore.add_texts(texts, metadatas=metadatas, embedding=embeddings)

# 벡터스토어에서 텍스트, 임베딩 및 메타데이터를 기반으로 Weaviate 인스턴스 생성
vectorstore = Weaviate.from_texts(
    texts,
    embeddings,
    metadatas=metadatas,
    client=client,
)

# 챗봇 모델 설정 (ChatOpenAI 사용)
llm = ChatOpenAI()

# 검색 엔진 설정 및 작동 방식 정의
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
)

# 여러 도구 로드
tools = load_tools(['wikipedia'], llm=llm)

# 에이전트 초기화
agent = initialize_agent(
tools,
llm,
agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
verbose=True
)

# 대화 메모리 버퍼 설정
memory = ConversationBufferMemory(memory_key='chat_history')

# 챗봇 플래너 및 실행기 로드
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

# PlanAndExecute 에이전트를 생성, 실행하여 사용자 질문에 답변 생성
agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True,
    reduce_k_below_max_tokens=True
)

# 사용자 입력을 받아서 실행하는 반복문
while True:
    user_input = input("prompt: ")  # 사용자로부터 입력을 받습니다.
    
    if user_input.lower() == "exit":  # 종료하려면 'exit'를 입력하면.
        break
    
    result = agent.run(user_input)  # 사용자의 입력을 agent.run에 전달하고 결과를 받습니다.
    print(result['response'])  # 결과를 출력합니다.
