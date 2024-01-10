import argparse
import os
import boto3
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('../.env')) # read local .env file

AWS_ACCESS_KEY = var = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME')
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')

# initialize pinecone client and embeddings
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_store = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)

s3 = boto3.client('s3', 
                  aws_access_key_id=AWS_ACCESS_KEY, 
                  aws_secret_access_key=AWS_SECRET_KEY,
                  region_name=AWS_REGION)


local_directory = 'tmp_pdfs'

""" 
Recursively search through a directory tree, filtering for pdf files.
For each file, create a LangChain PyPDFLoader, create a list of documents
using load_and_split with a RecursiveCharacterTextSplitter, and load the
docs into Pinecone
"""
def load_pdfs():

    """
    List and download all PDF files from the bucket
    """
    def list_files():
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET_NAME):
            for obj in page.get('Contents', []):
                file_name = obj['Key']
                if file_name.endswith('.pdf'):
                    local_file_path = os.path.join(local_directory, file_name)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    s3.download_file(BUCKET_NAME, file_name, local_file_path)
                    print(f'Downloaded {file_name} to {local_file_path}')

    list_files()

    for root, dirs, files in os.walk(local_directory):
        for file_name in files:
            if file_name.endswith('.pdf'):
                # print('Found a PDF')
                file_path = os.path.join(root, file_name)
                """
                Generic PyPDFLoader to load PDFs and split into documents.
                Note that this is only useful for PDFs that are already text-based.
                PDFs with images or tables will not be processed correctly.
                """
                loader = PyPDFLoader(file_path=file_path)
                docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
                print(f"loading {file_name} into pinecone index")
                # Customizing metadata to include bucket name and source key
                # This will allow RAG apps to retrieve the original file from S3
                for i in range(len(docs)):
                    dm = docs[i].metadata
                    dm['source'] = '/'.join(dm['source'].split('/')[1:])
                    dm['bucket'] = BUCKET_NAME
                    docs[i] = Document(page_content=docs[i].page_content, metadata=dm)               

                vector_store.add_documents(docs)
                # now delete the local file to clean up after ourselves
                os.remove(file_path)


"""
Main function to call load_pdfs. 
Specific bucket_name in .env file
"""
def main():
    load_pdfs()

if __name__ == "__main__":    
    main()