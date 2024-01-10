import streamlit as st
from dotenv import load_dotenv
import pickle
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    

def main():
    st.header("Chat with pdf")
    
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

# upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf') 
    query = st.text_input("Ask questions about your PDF file:")

    
    
    #st.write(pdf)
    if pdf is not None:
        vdb = None
        #pdf reader
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
        #     st.write('Embeddings Loaded from the Disk')
        
 
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        st.write("Embeddings Loaded")
        # with open(f"{store_name}.pkl", "wb") as f:
        #     pickle.dump(VectorStore, f)

        # Accept user questions/query
        # query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            st.write(query)
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(openai_api_key=openai_api_key)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()

