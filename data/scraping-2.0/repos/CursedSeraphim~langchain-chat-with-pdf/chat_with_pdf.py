import os

import streamlit as st
from dotenv import load_dotenv
from langchain import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase

def main():
    st.title("Chat with your PDF 💬")
    
    model_name = st.selectbox("Select the LLM Model:", ('gpt-4', 'gpt-3.5-turbo'))
    
    search_query = st.text_input('Enter a query to search the PDF')
    llm_query = st.text_input('Ask a question to the LLM')
    
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        knowledgeBase = process_text(text)
        
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        docs = None  # Initialize docs to None to use it later
        if search_query or llm_query:  # Trigger PDF search if either query exists
            docs = knowledgeBase.similarity_search(search_query if search_query else llm_query)
        
        if llm_query:  # Trigger LLM search only if llm_query exists
            llm = OpenAI(model_name=model_name)
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=llm_query)
                print(cost)
                
            st.write(response)
            st.write(docs)

if __name__ == "__main__":
    main()
