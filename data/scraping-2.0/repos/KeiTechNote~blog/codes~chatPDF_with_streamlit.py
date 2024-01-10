# -*- coding:utf-8 -*-
#
# chatPDF 를 제공하는 웹 서비스 만들기
# 
import os
import tempfile
import traceback

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

def uploaded_file_to_docs(uploaded_file):
    pages = None
    try:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_filepath = os.path.join(tmp_dir.name, uploaded_file.name)
        with open(tmp_filepath, "wb") as fp:
            fp.write(uploaded_file.getvalue())
        
        loader = PyMuPDFLoader(tmp_filepath)
        pages = loader.load()

    except:
        traceback.print_exc()
    
    finally:
        return pages

def generate_db(pages):
    db = None
    try:
        # Text Splitter 를 지정하고 페이지를 분할합니다. 
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 300,
            chunk_overlap  = 20,
            length_function = len,
            is_separator_regex = False,
        )
        texts = text_splitter.split_documents(pages)
        
        # Embedding 으로 Vector를 만듭니다. 
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")

        # Vector 를 DB 에 적재합니다.
        db = Chroma.from_documents(texts, embedding_function, persist_directory=r".\security\aws.db")
    except:
        traceback.print_exc()

    finally:
        return db


def main():
    # 제목
    st.title("chatPDF")
    st.write("---")

    # PDF 파일에서 페이지를 추출합니다.
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        pages = uploaded_file_to_docs()
    
        # DB 를 생성합니다.
        db = generate_db(pages)

        # 질문란을 만듭니다. 
        st.header("PDF에게 질문하세요")
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q8_0.bin", 
            model_type="llama"
        )

        question = st.text_input("질문을 입력하세요.")
        if st.button("질문하기"):
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            answer = qa_chain({"query": question})
            st.write(answer)


if __name__ == "__main__":
    main()
