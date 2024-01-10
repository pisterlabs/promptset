from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import sys
import logging
import os

key = '<YOUR OPENAI_API_KEY>'
os.environ["OPENAI_API_KEY"] = key
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def read_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            logging.info(f"loading {file_path}")
            try:
                loader = PyPDFLoader(file_path)
                document = loader.load_and_split()
                documents.extend(document)
            except Exception as e:
                logging.error(f"failed to read {file_path}", e)

    return documents


def create_index(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    qdrant = Qdrant.from_documents(
        documents, embeddings,
        path="./qdrant.vdb",
        collection_name="handbook",
    )
    return qdrant


if __name__ == '__main__':
    directory = './handbook/'
    documents = read_documents(directory)
    create_index(documents)
