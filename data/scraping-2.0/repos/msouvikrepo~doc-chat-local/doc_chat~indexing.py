import pinecone

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone

directory = (
    "Directory pointing to your docs"
)


def load_docs(directory):
    loader = TextLoader(directory)
    documents = loader.load()
    return documents


documents = load_docs(directory)


def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs_split = text_splitter.split_documents(documents)
    return docs_split


docs = split_docs(documents)

embeddings = SentenceTransformerEmbeddings(model_name="google/flan-t5-base") #Create embeddings with your favourite model

pinecone.init(api_key="Pinecone api key", environment="Pinecone environment")

index_name = "Pinecone index name"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)