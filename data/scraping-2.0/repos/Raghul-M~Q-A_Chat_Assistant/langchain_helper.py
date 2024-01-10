from langchain.llms import GooglePalm
import streamlit as st
import langchain
import sentence_transformers
import google.generativeai
import tqdm as notebook_tqdm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

api_key = ""
llm = GooglePalm(google_api_key=api_key,temperature=0.7)

#loader
loader = CSVLoader(file_path="Mental_Health_FAQ.csv",source_column="Questions")
docs=loader.load()

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path="faiss_index"


def create_vector_db():
    loader = CSVLoader(file_path="data.csv",source_column="prompt")
    docs=loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_thresold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "Answers" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
        
        CONTEXT: {context}
        
        QUESTION: {question}"""
        
        
    PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs
                                           )
    return chain



if __name__ == "__main__":
    create_vector_db(vectordb_file_path,instructor_embeddings)

