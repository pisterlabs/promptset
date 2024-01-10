# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import sqlite3
import tempfile
import streamlit as st
import modules

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
import config

OPENAI_API_KEY = config.OPENAI_API_KEY

# 라마, gpt3.5, gpt4 중 택 1 옵션 넣기 < 해결
# 여러 pdf문서 넣을 수 있게 하기(프레스바 넣기)
# 임베딩된 값들 저장하고 다시 불러올 수 있게 하기
# 전체 답변과
# 각 출처에서 추출하여 같이 제공하기

# 제목
st.title("SearchingPDFs")
st.write("---")


# 파일 업로드
pdf_files = st.file_uploader("PDF 파일을 올려주세요!", accept_multiple_files=True, type=['pdf'])
st.write("---")
chat = modules.Chat_UI(is_debugging=False)#채팅 모듈

select_model = st.radio(
    "Select to use Model",
    ["LLaMA", "GPT3.5", "GPT4"],
    index=1,
    disabled=True
)

st.write("You selected:", select_model)


def read_pdfs(pdf_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        pages = []
        for file in pdf_files:
            temp_filepath = os.path.join(temp_dir, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            for page in loader.load_and_split():
                pages.append(page)
    return pages


def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)
    return texts


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", update_interval=2):
        self.container = container
        self.text = initial_text
        self.token_buffer = []
        self.update_interval = update_interval  # 토큰이 몇 개 모일 때 UI를 업데이트할지 결정

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_buffer.append(token)
        
        # 버퍼에 저장된 토큰의 수가 update_interval에 도달하면 UI를 업데이트
        if len(self.token_buffer) >= self.update_interval:
            self.text += ''.join(self.token_buffer)
            print(self.token_buffer)
            self.container.markdown(self.text)
            self.token_buffer = []

    def on_llm_end(self, response, **kwargs):
        chat.get_bot_answer(self.text)
        self.text = ""

    def on_llm_error(self,error):
        print(f"error occured: {error}")

def Answer_Bot():
    user_input = chat.on_input_change()
    stream_handler = StreamHandler(st.empty())

    if select_model == "LLaMA":
        select_model_name = ""
    elif select_model == "GPT3.5":
        select_model_name = "gpt-3.5-turbo"
    elif select_model == "GPT4":
        select_model_name = ""

    llm = ChatOpenAI(
        model_name= select_model_name,
        temperature=0,
        streaming=True,
        openai_api_key=OPENAI_API_KEY,
        callbacks=[stream_handler])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)

    response = qa_chain(user_input)
    for source in response['source_documents']:
        file_name = os.path.basename(source.metadata['source'])
        page_number = source.metadata['page']
        # chat.get_bot_answer(f"file_name: {file_name} , page: {page_number}")
        # print('file_name: ', file_name, 'page: ', page_number)


if pdf_files:

    # Load
    pages = read_pdfs(pdf_files)

    # Split
    texts = split_pages(pages)
    
    # Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Save it into Chroma
    persist_directory = None  # "./data_store"
    db = Chroma.from_documents( #chromadb 임베딩된 텍스트 데이터들을 효율적으로 저장하기위한 모듈
        documents=texts,
        embedding=embeddings_model,
        persist_directory=persist_directory)

    # Set retriever and LLM
    retriever = db.as_retriever()

    # Question
    st.header("KnowBot!!")

    
    
    chat.display_chat()
    with st.container():
        st.text_input("User Input:", on_change=Answer_Bot, key="user_input")



