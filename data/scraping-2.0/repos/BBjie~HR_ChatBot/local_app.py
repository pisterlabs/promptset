from flask import Flask, request, jsonify, render_template
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from config import configs
import time



# Initialize Embeddings and Load Documents
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create Vector Store
vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/pet_cosine")
print("Vector Store Created.......")

# Initialize LLM and other components
local_llm = "mistral-7b-instruct-v0.1.Q8_0.gguf"
config = configs
llm = CTransformers(model=local_llm, model_type="mistral", lib="avx2", **config)
print("LLM Initialized....")

# Set up the prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Load the vector store as a retriever
retriever = vector_store.as_retriever(search_kwargs={"k":1})

chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=retriever, 
    chain_type_kwargs = {"prompt": prompt})

if __name__ == "__main__":
    while True:
        question = input("Ask a question (type 'exit' to quit): ")
        if question.lower() in ['exit', 'quit']:
            break

        start_time = time.time()  # Start timing

        answer = chain.run({"query": question})

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print("Answer:", answer)
        print("Response Time:", elapsed_time, "seconds")
