import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI,HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
os.environ['HUGGINGFACEHUB_API_TOKEN']='hf_WOVaQPBTHaOzYoUrfRIczAqBBaTGrkCvrz'

def main():
    st.header('Chat with PDF document')
    st.sidebar.title('LLM ChatApp using LangChain')
    st.sidebar.markdown('''
    This is an LLM powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://openai.com/docs/models) LLM Model
    ''')

    #Upload a PDF File

    pdf = st.file_uploader("upload your PDF File",type='pdf')
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #st.write(text)
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            strip_whitespace=''
        )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks[0])

        download_name = pdf.name[:-4]

        faiss_index_path = f"{download_name}.index"


        if os.path.exists(faiss_index_path):
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            VectorStore= FAISS.load_local(faiss_index_path,embeddings)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            VectorStore = FAISS.from_texts(chunks, embeddings)
            VectorStore.save_local(faiss_index_path)
            st.write('Embeddings Created')

        query = st.text_input("Enter your question from your PDF File")

        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            #llm = OpenAI()
            llm = HuggingFaceHub(repo_id = 'google/flan-t5-large',model_kwargs={"temperature":0,"max_length":64})
            #llm = HuggingFaceHub(model_kwargs={"temperature":0,"max_length":64})
            chain = load_qa_chain(llm = llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()