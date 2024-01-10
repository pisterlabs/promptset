import os
import json
import sys
from typing import Iterable

from dotenv import load_dotenv, find_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

from langchain.schema import Document
from langchain.document_loaders.pdf import DocumentIntelligenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import openai
import shutil

# Load environment variables
_ = load_dotenv(find_dotenv())

# Set up Azure and OpenAI credentials
azure_endpoint = os.environ['AZURE_COGSRVC_ENDPOINT']
azure_key = os.environ['AZURE_COGSRVC_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']

# Define functions
def save_docs_to_jsonl(array: Iterable[Document], file_path: str) -> None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path: str) -> Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

def process_directory(directory: str) -> None:
    # Create DocumentAnalysisClient
    document_analysis_client = DocumentAnalysisClient(
        endpoint=azure_endpoint, credential=AzureKeyCredential(azure_key)
    )

    # Loop through PDF files in directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(directory, file_name)
            json_path = pdf_path + '.json'

            # Check if JSON file already exists
            if os.path.exists(json_path):
                continue

            # Load PDF file
            loader = DocumentIntelligenceLoader(
                    directory + "/" + file_name,
                    client=document_analysis_client,
                    model="prebuilt-document") # e.g. prebuilt-document

            document = loader.load()

            # Submit document to DocumentAnalysisClient
            result = document_analysis_client.analyze_document(document=document)

            # Save result to JSON file
            with open(json_path, 'w') as json_file:
                json.dump(result, json_file)

def load_json_files(directory: str):
    # Loop through JSON files in directory
    result = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.json'):
            json_path = os.path.join(directory, file_name)
            result.append(load_docs_from_jsonl(json_path))
    return result

reprocess_files = False # Set to True to reprocess files by submitting pdfs to Azure Cognitive Services

if reprocess_files:
    # Scan directory for PDF files and process them with DocumentAnalysisClient
    # If a JSON file already exists for a PDF file, skip it
    
    process_directory('docs')

    all_documents = load_json_files('docs')

    # Split documents into chunks
    chunk_size =1000
    chunk_overlap = 200

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = []

    for document in all_documents:
        document_splits = r_splitter.split_documents(document)
        splits.extend(document_splits)

    # Create embedding
    embedding = OpenAIEmbeddings()
    persist_directory = 'docs/chroma/'

    # Delete directory if it already exists (to avoid errors)
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        shutil.rmtree(persist_directory)

    # Create vector database, retrieve embeddings, save them to vector database
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    # Save vector database to disk
    vectordb.persist()

llm_name = "gpt-3.5-turbo"

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

question = "How do I turn on AC for the back row?"
docs = vectordb.similarity_search(question,k=5)

print("Question: " + question)

llm = ChatOpenAI(model_name=llm_name, temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
result = qa_chain({"query": question})
print("-----------------------------------------------------------------------------")
print("Answer: " + result["result"])

