# ======================================================================================================================
# 챗GPT 연동 서비스 모듈
# 출처: https://github.com/jaysooo/chatgpt_streamlit_app
# [ ChatGPT API를 활용한 챗봇 서비스 구축 프레임워크 ]
# 1) LangChain(*) - 현재소스에서 사용
# 2) Semantic Kernel
#
# [ 성능향상을 위한 방법 ]
# 1) RAG를 사용하여, 문서를 인베딩벡터로 변환하고, 유사도 검색을 RAG를 사용하여 처리하는 방식(*) <- 현재소스에서 사용
# ┌-----------------┐   HuggingFaceEmbeddings   ┌-----------------┐         ┌--------------┐
# | 관련문서(pdf)   ├-------------------------->| 인베딩벡터 변환 ├-------->| Chroma DB    |
# └-----------------┘                           └-----------------┘         └--------┬-----┘
# ┌--------------------┐                        ┌------------------┐                 | 질문과 유사한 문서를 검색
# | 추출된 내용 +      |<-----------------------┤ 연관된 내용 추출 |<----------------┛
# | 질문 내용          |                        └------------------┘
# └---------┬----------┘
#           |
# ┌---------∨----------┐                        ┌------------------┐
# | LangChain          ├----------------------->| ChatGPT 3.5      |
# └--------------------┘                        └------------------┘
# 2) 웹 서치 연동하기
#   - Google Custom Search API
#   - Serper API
#   - Serp API
# 3) 가장 접합한 답변 선택하기
#
# [ 추가해야 할 기능들 ]
# [X] 파일 업로드 기능
# [X] 업로드된 파일에 대해 인베딩벡터로 변환하고, Chroma DB를 다시 생성하는 기능 <- 추가된 파일만 처리하도록 변경 필요
# [X] 화면에서 GPT 모델, 온도, 토큰수를 선택할 수 있도록 기능 추가
# [ ] 추가된 파일만 임베팅벡터 처리하도록 변경
# ----------------------------------------------------------------------------------------------------------------------
# 2023.09.07 - 초기모듈 작성
#            - 인베딩벡터 정보를 파일로 저장하고, 유사도 검색을 FAISS를 사용하여 처리하는 방식 사용
# 2023.09.08 - Chroma DB 생성 및 로딩 함수로 변경
#            - 서비스 흐름도 작성
# 2023.09.10 - GPT 모델, 온도를 선택하여 챗팅할 수 있도록 기능 추가
# ======================================================================================================================
from dotenv import load_dotenv
from chatgpt_logger import logger
import openai
import os
import shutil

# ----------------------------------------------------------------------------------------------------------------------
# 환경 변수들을 딕셔너리로 묶어서 반환한다.
# - 현재는 사용하고 있지 않지만 향후 확장성을 위해 기본구조를 만들어 놓음
# ----------------------------------------------------------------------------------------------------------------------
def get_openai_options():
    openai_model = os.environ.get("OPENAI_MODEL")
    openai_temperature = os.environ.get("OPENAI_TEMPERATURE")
    oepnai_max_token =os.environ.get("OPENAI_MAX_TOKEN") 

    args = {
        'model': openai_model,
        'temperature' : openai_temperature,
        'max_token' : oepnai_max_token,
    }

    return args

# ----------------------------------------------------------------------------------------------------------------------
# 환경 변수를 로딩한다.
# ----------------------------------------------------------------------------------------------------------------------
def load_env():

    # set environment for application
    load_dotenv()
    version = os.environ.get("VERSION")
    openai_token = os.environ.get("OPENAI_TOKEN")
    huggingfacehub_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    version = os.environ.get("VERSION")

    os.environ["OPENAI_API_KEY"] = openai_token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_token
    # os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # set openai connection
    openai.api_key = openai_token

    logger.info(f"app version :  {version} \t")

# ----------------------------------------------------------------------------------------------------------------------
# 해당 디렉토리의 인베팅된 문서들로부터 Chroma DB로 생성한다.
# ----------------------------------------------------------------------------------------------------------------------
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
persist_directory = "./chroma_db"
# 2023.09.07 - (임시조치) 아래와 같이 try except 처리를 해주지 않으면, 아래와 같은 에러가 발생한다.
#            - 이유는 아직 모르겠고, Chroma DB가 사용하는 의존성 모듈들의 버전이 맞지 않는 것 같음
# InvalidInputException: Invalid Input Error: Required module 'pandas.core.arrays.arrow.dtype' failed to import,
# due to the following Python exception: ModuleNotFoundError: No module named 'pandas.core.arrays.arrow.dtype'
# (조치내용)
# 2023.09.09 - 가상화환경을 기존 Python 3.9 -> 3.10로 업그레이드 시 해결됨
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# ----------------------------------------------------------------------------------------------------------------------
# 챗봇 서비스 API
# ----------------------------------------------------------------------------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
def answer_from_chatgpt(query, model_name, temperature):
    # model_name = "gpt-3.5-turbo"
    # temperature = 0.3
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    # chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    # matching_docs = db.similarity_search(query)
    # answer = chain.run(input_documents=matching_docs, question=query)


    # RetrieverQA Chain 사용하기
    # 체인타입(chain_type)
    # - stuff: 문서 검색 결과를 하나의 텍스트로 합치고, 그 텍스트를 답변 생성에 사용, 가장 간단하고 빠른 체인
    # - map_reduce: 문서 검색 결과를 각각 답변 생성에 사용하고, 그 결과들을 점수화하고,
    #               가장 높은 점수를 가진 답변을 선택하는 체인, 가장 정확하고 다양한 답변을 생성할 수 있는 체인
    # - map_reduce_with_context: map_reduce와 비슷하지만, 문서 검색 결과를 답변 생성에 사용할 때,
    #               문서의 제목과 URL을 함께 넘겨주는 체인, 문서의 출처와 관련성을 고려할 수 있는 체인
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
    answer = retrieval_chain.run(query)

    return answer

# ----------------------------------------------------------------------------------------------------------------------
# 문서들을 인베팅벡터로 변환하고, Chroma DB를 생성한 후 저장한다.
# - 서비스가 시작될 때마다 문서변환을 하지 않고, 미리 변환된 문서를 로딩하여 사용할 수 있도록 함수를 분리함
# [ Open Source Vector DB ]
# 출처: https://blog.futuresmart.ai/using-langchain-and-open-source-vector-db-chroma-for-semantic-search-with-openais-llm
# > conda install -c conda-forge chromadb
# ----------------------------------------------------------------------------------------------------------------------
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

def doc_to_chroma(directory):
    # 해당 디렉토리의 모든 pdf 파일을 읽어서, 문서를 로딩한다.
    directory = './pdf'
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # len(documents)

    # Chroma DB를 해당 디렉토리에 저장하기 전에 이전 데이터를 삭제한다.
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # 문서를 적당한 크기로 쪼개서, 문서를 분할한다.
    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    # print(len(docs))

    # # 문서를 벡터로 변환하고, Chroma DB 객체를 생성한다.
    # # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = HuggingFaceEmbeddings()
    # db = Chroma.from_documents(docs, embeddings)

    # Chroma DB를 해당 디렉토리에 저장한다.
    persist_directory = "chroma_db"
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()

    # Chroma DB를 다시 로딩한다.
    global db
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


