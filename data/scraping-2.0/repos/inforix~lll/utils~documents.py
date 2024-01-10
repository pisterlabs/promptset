from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from typing import List
from langchain.docstore.document import Document

def load_documents(path:str = "./data/", pattern:str = "**/*.txt") -> List[Document]:
  loader = DirectoryLoader(path, pattern)
  documents = loader.load()
  text_splitter = CharacterTextSplitter(        
   chunk_size = 1000,
   chunk_overlap  = 100,
  )
  texts = text_splitter.split_documents(documents)
  return texts