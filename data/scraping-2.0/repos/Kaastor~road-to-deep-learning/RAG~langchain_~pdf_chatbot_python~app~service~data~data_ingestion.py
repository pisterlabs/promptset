import os
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_.pdf_chatbot_python.app.service.constants import PINECONE_API_KEY, PINECONE_ENVIRONMENT, \
    PINECONE_INDEX_NAME, PINECONE_NAME_SPACE, OPENAI_API_KEY

filePath = os.getcwd()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def init_pinecone():
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )


def run():
    try:
        # Load the data
        loader = PyPDFDirectoryLoader(filePath)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        raw_docs = loader.load_and_split(text_splitter=text_splitter)

        # Create and store the embeddings in the vectorStore
        init_pinecone()
        embeddings = OpenAIEmbeddings()
        Pinecone.from_documents(raw_docs, embeddings,
                                index_name=PINECONE_INDEX_NAME,
                                namespace=PINECONE_NAME_SPACE)
    except Exception as error:
        print('error', error)
        raise Exception('Failed to ingest your data')


if __name__ == '__main__':
    run()
