import os
import streamlit as st

from langchain import Cohere
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

ignore_warnings = True

os.environ["COHERE_API_KEY"] = ""

llm = Cohere(model="command-nightly", temperature=0)

# Embedding
embeddings = CohereEmbeddings(model="multilingual-22-12")


def upload_sec_file_to_vector_db(fileUrl):
    loader = PyPDFLoader(fileUrl)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f"# of documents = {len(docs)}")
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return retriever


def query_vector_store(retriever, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    result = qa({"query": query})
    return result


def summerise_large_pdf(fileUrl):
    loader = PyPDFLoader(fileUrl)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(texts)


stocks = {
    "Alphabet - 'GOOG'": {"name": "Alphabet Inc.", "symbol": "GOOG", "cik": "0001652044",
                          "url": "https://abc.xyz/assets/a7/5b/9e5ae0364b12b4c883f3cf748226/goog-exhibit-99-1-q1-2023"
                                 "-19.pdf",
                          "ten_k_url": "https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf"},
    "Apple - 'AAPL'": {"name": "APPLE INC", "symbol": "AAPL", "cik": "0000320193",
                       "url": "https://www.apple.com/newsroom/pdfs/FY23_Q2_Consolidated_Financial_Statements.pdf",
                       "ten_k_url": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/b4266e40-1de6-4a34-9dfb"
                                    "-8632b8bd57e0.pdf"}
}

# page construction
st.set_page_config(page_title="Relationship Manager Investment Dashboard ABC Plc", layout="wide",
                   initial_sidebar_state="collapsed")

col1, col2 = st.columns((1, 3))

st.title("Relationship Manager Investment Dashboard ABC Plc")

selected_stock = col1.selectbox("Select a stock", options=list(stocks.keys()))

selected_stock_name = stocks[selected_stock]["name"]
selected_stock_url = stocks[selected_stock]["url"]
selected_stock_ten_k_url = stocks[selected_stock]["ten_k_url"]

col2.subheader("Summary of Last Quarter Financial Performance")
col2.write(summerise_large_pdf(selected_stock_url))

vector_store_hook = upload_sec_file_to_vector_db(selected_stock_ten_k_url)

param_query_1 = f"What are the key products and services of {selected_stock_name} ?"
param_query_2 = f"What are the new products and growth opportunities for {selected_stock_name} ?"
param_query_3 = f"What are the key strengths of {selected_stock_name} ?"
param_query_4 = f"What are the key competitors of {selected_stock_name} ?"
param_query_5 = f"What are the principal threats to {selected_stock_name} ?"
param_query_6 = f"What are the key risks to {selected_stock_name} ?"
param_query_7 = f"What are the key opportunities for {selected_stock_name} ?"
param_query_8 = f"What are the key challenges for {selected_stock_name} ?"

col2.subheader("Summary of Last Year Financial Performance")
col2.write(query_vector_store(vector_store_hook, param_query_1))
col2.write(query_vector_store(vector_store_hook, param_query_2))
col2.write(query_vector_store(vector_store_hook, param_query_3))
col2.write(query_vector_store(vector_store_hook, param_query_4))
col2.write(query_vector_store(vector_store_hook, param_query_5))
col2.write(query_vector_store(vector_store_hook, param_query_6))
col2.write(query_vector_store(vector_store_hook, param_query_7))
col2.write(query_vector_store(vector_store_hook, param_query_8))

col2.subheader("Chat With Last Year Financial Performance !")
query = col2.write("Please Enter Your Query in Plain Text ! ")
result = query_vector_store(vector_store_hook, query)
col2.write(result)