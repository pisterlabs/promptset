import os
import pinecone

from langchain.vectorstores import Pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .llm import get_embedding_llm
from dotenv import load_dotenv

load_dotenv()


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
embedding_llm = get_embedding_llm()


def ingest_docs():
    # reset
    # index = pinecone.Index(INDEX_NAME)
    # index.delete(delete_all=True)

    docs = ingest_pdf("docs/uob.pdf")
    print(f"Adding {len(docs)} vectors to Pinecone")
    Pinecone.from_documents(docs, embedding_llm, index_name=INDEX_NAME)

    urls = ['https://www.channelnewsasia.com/', 'https://www.straitstimes.com', 'https://www.theverge.com/', "https://www.theverge.com/2023/11/20/23968829/microsoft-hires-sam-altman-greg-brockman-employees-openai"]
    docs = ingest_url(urls)
    print(f"Adding {len(docs)} vectors to Pinecone")
    Pinecone.from_documents(docs, embedding_llm, index_name=INDEX_NAME)

    print("****Loading to vectorestore done****")


def ingest_pdf(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    return get_splitted_text(loader)


def ingest_url(url):
    loader = WebBaseLoader(url)
    return get_splitted_text(loader)


def get_splitted_text(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=30, separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents=documents)
    return docs


if __name__ == "__main__":
    ingest_docs()
