import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain


import os
load_dotenv()

embeddings  = OpenAIEmbeddings()
 
def create_db(content) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(text = content)
    print(*docs)

    db = FAISS.from_texts(docs, embeddings)
    return db

def ans_query(db,query,k=4):

    relevant_docs = db.similarity_search_with_score(query, k=k)
    
    most_relevant_index = min(range(len(relevant_docs)), key=lambda i: relevant_docs[i][1])
    most_relevant_doc, most_relevant_score = relevant_docs[most_relevant_index]
    most_relevant_page_content = most_relevant_doc.page_content
    
    relevant_doc_content = " ".join([doc.page_content for doc, _ in relevant_docs])
    

    llm = OpenAI(model_name="text-davinci-003")
    prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            You are a helpful assistant that that can answer questions about the document 
            based on the document content.
            
            Answer the following question: {question}
            By searching the following document content: {docs}
            
            Only use the factual information from the document to answer the question.
            
            If you feel like you don't have enough information to answer the question, say "I don't know".
            
            Your answers should be verbose and detailed.
            """,
        )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=relevant_doc_content)
    response = response.replace("\n", "")
    return response,most_relevant_page_content


    






