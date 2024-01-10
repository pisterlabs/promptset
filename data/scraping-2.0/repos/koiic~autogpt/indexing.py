from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone

import pinecone


pinecone.init(
    api_key="",
    environment="us-west1-gcp",
    
)

index_name = "langchain_chatbot"


embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

directory = 'venv/lib'

def load_doc(directory):
    loader = DirectoryLoader(directory)
    return loader.load()

def split_documents(documents, chunk_size=100, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs =  text_splitter.split(documents)
    return docs

def get_similar_documents(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

documents = load_doc(directory)
len(documents)
docs = split_documents(documents)

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


