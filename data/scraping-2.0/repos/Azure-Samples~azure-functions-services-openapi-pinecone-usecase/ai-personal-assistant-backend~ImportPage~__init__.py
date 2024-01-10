import os
import logging
import pinecone
from dotenv import load_dotenv
import azure.functions as func
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

index_name = 'functions'

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

def main(req: func.HttpRequest) -> func.HttpResponse:

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        url = req_body.get('url')

    if url:
        logging.info(f"Retrieving: {url}")
        loader = UnstructuredURLLoader(urls=[url])
        document = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(document)

        logging.info(f"Split into {len(docs)} chunks")
        embeddings = OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002"
        )
        load_dotenv()
        embeddings.openai_api_base = os.getenv("OPENAI_API_BASE")
        embeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
        embeddings.openai_api_version = os.getenv("OPENAI_API_VERSION")
        embeddings.openai_api_type = os.getenv("OPENAI_API_TYPE")

        logging.info(f"Embeddings initialized {os.getenv('OPENAI_API_BASE')}")
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        logging.info(f"Indexed {len(docs)} chunks")
        index_description = pinecone.describe_index(index_name)
        logging.info(index_description)

        return func.HttpResponse(f"Indexed {len(docs)} chunks", status_code=200)
    else:
        return func.HttpResponse(
             "Pass a json object with a url property",
             status_code=400
        )
