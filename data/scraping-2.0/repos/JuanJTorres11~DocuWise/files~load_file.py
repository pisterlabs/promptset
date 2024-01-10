from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os
import pinecone

load_dotenv()

print("Loading file...")
loader = PyMuPDFLoader("./resolucion-3280-de-2018.pdf")
data = loader.load()

print("File loaded.")

print("Splitting text...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 500,
    length_function = len,
    is_separator_regex = False,
)
docs = text_splitter.split_documents(data)


print("Text splitted.\n")


model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Loading embeddings model...")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
)

index_name = "documents-index"

print("initializing pinecone...")

docsearch = Pinecone.from_documents(docs, embeddings_model, index_name=index_name)

print("pinecone initialized.")


