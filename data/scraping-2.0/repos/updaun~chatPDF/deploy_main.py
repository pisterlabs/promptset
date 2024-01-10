__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# 제목
st.title("ChatPDF")
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 파일 업로드
uploaded_file = st.file_uploader("Choose a file", type="pdf")
st.write("---")

# 업로드 된 파일이 있으면
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    
    # Loader
    # loader = PyPDFLoader("unsu.pdf")
    # pages = loader.load_and_split()

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # Embeddings
    embeddings_model = OpenAIEmbeddings()

    # Chroma
    vectordb = Chroma.from_documents(texts, embeddings_model)

    # load from directory
    # vectordb = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")
    # vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

    # Question
    # question = "아내가 먹고 싶어하는 음식은 무엇이야?"
    
    # 관련 문서를 찾는다.
    # llm = ChatOpenAI(temperature=0)
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=vectordb.as_retriever(), llm=llm
    # )
    # docs = retriever_from_llm.get_relevant_documents(query=question)

    st.header("ChatPDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요.")
    st.write("---")

    # 결과를 보여준다.
    if st.button("질문하기"):
        with st.spinner("waiting..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
