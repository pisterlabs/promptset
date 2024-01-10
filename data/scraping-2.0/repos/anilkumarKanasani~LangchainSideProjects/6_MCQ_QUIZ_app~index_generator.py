from environs import Env
env = Env()
env.read_env()

from langchain.document_loaders import PyPDFDirectoryLoader

#Loading Documnets from Source
def load_docs(directory):
  loader = PyPDFDirectoryLoader(directory)
  documents = loader.load()
  return documents

directory = '../data/Docs/'
documents = load_docs(directory)
print('Number of documents at source level : ', len(documents))


#Splitting Documnets into small chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_docs(documents, chunk_size=600, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

split_docs = split_docs(documents)
print('Number of documents after splitting : ', len(split_docs))


#Embedding model preparion
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(deployment=env("AZURE_EMBEDDING_DEPLOYMENT"))


# testing the embed model
query_result = embeddings.embed_query("Hello Buddy")
print('Dimension of the embed vector : ', len(query_result))

# preparing vector store and index
import pinecone
from langchain.vectorstores import Pinecone
from environs import Env
env = Env()
# Read .env into os.environ
env.read_env()
pinecone.init(
    api_key=env("PINECONE_API_KEY"),
    environment="eastus-azure"
)

index_name = "mcq-creator"

index = Pinecone.from_documents(split_docs, embeddings, index_name=index_name)
print(index_name , ' index is updated with latest information in Pinecone')