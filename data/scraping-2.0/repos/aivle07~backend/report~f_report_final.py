from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

# 금융 리포트 PDF 읽는 함수
def get_pdf_report(openai_api_key, file):
    # PDF 파일 업로드
    #loader = PyPDFLoader("fin2.pdf")
    loader = PyPDFLoader(file)
    # 페이지 로드 및 텍스트로 분할
    pages = loader.load_and_split()
    # 텍스트 문장 단위로 Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    # 페이지에서 텍스트 추출
    texts = text_splitter.split_documents(pages)
    # OpenAI의 Embeddings 모델을 사용하여 텍스트를 임베딩
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Chroma 데이터베이스에 문서를 로드
    db = Chroma.from_documents(texts, embeddings_model)
    return db

# PDF 리포트 질문에 대답하는 함수
def get_pdf_report_answer(db, question):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    result = qa_chain({"query": question})
    return result['result']

if __name__ == '__main__':
    openai_api_key = os.getenv("OPENAI_API_KEY")
    db = get_pdf_report(openai_api_key)
    question = input("질문을 입력하세요: ")
    result = get_pdf_report_answer(db, question)
    print(result)
    # llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
    # qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    # result = qa_chain({"query": question})
    # print(result['result'])
