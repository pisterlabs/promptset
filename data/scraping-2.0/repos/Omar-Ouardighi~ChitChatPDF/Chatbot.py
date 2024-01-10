import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import streamlit as st


class Chatbot:
    
    def __init__(_self, openai_api_key, temperture, model_name, chunk_size):
        _self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=100)
        _self.underlying_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        _self.llm = ChatOpenAI(model_name=model_name, temperature=temperture, openai_api_key=openai_api_key)

        qa_template = """
                Your role is to assist users by answering questions about PDFs they upload. You should focus on analyzing the content of these PDFs, 
                providing detailed and accurate answers to questions based on the information in the document. 
                It's important to handle a range of questions, from specific details to broader concepts covered in the PDF.

        context: {context}
        =========
        question: {question}
        ======
        """

        _self.QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])   

    def load_document(self, files):
        text=""
        for file in files:
            if file.name.endswith(".pdf"):
                pdf = PdfReader(file)
                for page in pdf.pages:
                    text += page.extract_text()
            elif file.name.endswith(".txt"):
                pdf = PdfReader(file)
                for page in pdf.pages:
                    text += page.extract_text()
        return text
    

    @st.cache_resource(show_spinner="creating a vectorstore...")
    def vectorize(_self, texts):
    
        chunks = _self.splitter.split_text(texts)
        vectorstore = Chroma.from_texts(chunks, _self.underlying_embeddings)

        return vectorstore
    
    def build_chain(_self, vectorstore):
        chain = ConversationalRetrievalChain.from_llm(llm = _self.llm, 
                                            retriever = vectorstore.as_retriever(search_kwargs={"k": 5}),
                                            combine_docs_chain_kwargs={'prompt': _self.QA_PROMPT},
                                            return_source_documents=True)
        return chain
