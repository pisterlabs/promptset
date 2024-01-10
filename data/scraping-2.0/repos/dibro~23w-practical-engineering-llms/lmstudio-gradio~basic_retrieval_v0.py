import os
import logging
import gradio as gr
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Setup logging
logging.basicConfig(level=logging.INFO)

# Configuration Variables
config = {
    "dirpath": './data/PDF/',
    "pdf_glob_pattern": "./*.pdf",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "qdrant_path": "./tmp/local_qdrant",
    "qdrant_collection_name": "pdfs",
    "api_url": 'http://localhost:1234/v1',
    "api_key": "na",
    "temperature": 0,
    "k_retrieval": 3
}

# Load and process documents
try:
    loader = DirectoryLoader(config['dirpath'], glob=config['pdf_glob_pattern'], loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])
    texts = text_splitter.split_documents(docs)
except Exception as e:
    logging.error(f"Document processing failed: {e}")
    raise

# Delete the storage.sqlite file
storage_file_path = os.path.join(config['qdrant_path'], "collection", config['qdrant_collection_name'], "storage.sqlite")
try:
    if os.path.exists(storage_file_path):
        os.remove(storage_file_path)
        logging.info(f"Deleted storage.sqlite for collection: {config['qdrant_collection_name']}")
    else:
        logging.info(f"No storage.sqlite file found for collection: {config['qdrant_collection_name']}")
except Exception as e:
    logging.error(f"Error occurred during file deletion: {e}")
    raise

# Vector database setup
try:
    qdrant = Qdrant.from_documents(documents=texts, embedding=GPT4AllEmbeddings(), path=config['qdrant_path'], collection_name=config['qdrant_collection_name'])
    retriever = qdrant.as_retriever(search_type='similarity', search_kwargs={'k': config['k_retrieval']})
except Exception as e:
    logging.error(f"Qdrant initialization failed: {e}")
    raise

# Template and Chat model setup
template = """Answer the question based only on the following context:
{context}
Question: 
{question}
ANSWER:
"""
prompt = PromptTemplate(input_variables=["question", "context"], template=template)
llm = ChatOpenAI(base_url=config['api_url'], api_key=config['api_key'], temperature=config['temperature'])

# Define QA Chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=False, chain_type_kwargs={"prompt": prompt})

def call_qa(question):
    try:
        response = qa_chain({"query": question})
        return response['result'].strip()
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface
demo = gr.Interface(fn=call_qa, inputs="text", outputs="text", title="QA System", description="Ask any question and get an answer.")
demo.launch(show_api=False)
