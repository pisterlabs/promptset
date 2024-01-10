from langchain.document_loaders import DirectoryLoader

# directory = r'C:\Users\hplap\OneDrive\Desktop\data'
directory = './ML_model/data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print(len(documents))


from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="59d1646a-e045-4da3-a6ab-5c515b990d29",  # find at app.pinecone.io
    # api_key="YOUR_PINECONE_API_KEY",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
    # environment="YOUR_PINECONE_ENV"  # next to api key in console
)
index_name = "langchain-chatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def get_similiar_docs(query,k=3,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs






