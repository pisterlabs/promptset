from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


async def create_memmory(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    new_memory = FAISS.from_documents(docs, embeddings).as_retriever()

    return new_memory


async def create_memmory2(text: str):
    loader = TextLoader(text)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    new_memory = FAISS.from_documents(docs, embeddings).as_retriever()

    return new_memory