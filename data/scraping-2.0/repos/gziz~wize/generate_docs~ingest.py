import tiktoken
from pathlib import Path
from tqdm.auto import tqdm

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import Chroma

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_text_splitter():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,  # number of tokens overlap between chunks
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )
    return text_splitter

def get_embeddings_engine():
    from langchain.embeddings import OpenAIEmbeddings

    embedding_engine = OpenAIEmbeddings(model="text-embedding-ada-002")
    return embedding_engine

DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma")

def load():
    text_splitter = get_text_splitter()
    embeddings = get_embeddings_engine()

    documents = []
    for file in DOCS_DIR.rglob("*.txt"):
        loader = TextLoader(str(file))
        pages = loader.load_and_split()
        documents += text_splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=f"chroma"
        )

    vectorstore.persist() # Make it persist in disk

load()

