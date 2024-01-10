
import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
)
INDEX_NAME = "donkey-betz"


def ingest_docs():
  loader = ReadTheDocsLoader(path="coding-docs", features='html.parser') 
  raw_doc = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
  documents = text_splitter.split_documents(documents=raw_doc)
  print(f"splitting 771 documents into  {len(documents)} chunks")
  
  for doc in documents:
    old_path = doc.metadata["source"]
    new_url = old_path.replace("langchain-docs", "https:/")
    doc.metadata.update({"source": new_url})
    
  print(F"Going to insert {len(documents)} to Pinecone")
  
  embeddings = OpenAIEmbeddings()
  Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
  
  print("****** Added to Pinecone Vectorstore vectors")





if __name__ == '__main__':
  ingest_docs()