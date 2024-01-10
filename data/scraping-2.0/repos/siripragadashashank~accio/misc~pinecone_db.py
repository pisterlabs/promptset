import os
import pinecone
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')


loader = DirectoryLoader(
    './raw',
    glob='./*.txt',
    loader_cls=TextLoader,
    show_progress=True
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs_split = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
print("init done")

doc_db = Pinecone.from_documents(
    docs_split,
    embeddings,
    index_name='accio'
)
print("index creation done")

query = "what is this legal agreement about"
docs = doc_db.similarity_search(query)
print(docs[0].page_content)


