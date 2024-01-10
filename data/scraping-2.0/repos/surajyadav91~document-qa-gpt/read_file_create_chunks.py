import tiktoken
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

class ReadCreateChunk:
    def __init__(self, config):
        self.chunk_size = config['chunk_size']
        self.overlap_chunk_size = config['overlap_chunk_size']
        self.file_path = config['file_path']
        
    def read_file(self, path):
        loader = PyPDFLoader(path)
        documents = loader.load()
        return documents
    
    def text_splitter(self, documents):
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap_chunk_size)
        split = text_splitter.split_documents(documents)
        return split
    
    def store_chunks(self, file_path):
        documents = self.read_file(file_path)
        chunks = self.text_splitter(documents)
        chunk_list = []
        
        for i in range(len(chunks)):
            chunk_list.append(chunks[i].page_content)

        with open('chunks.pkl', 'wb') as f:
            pickle.dump(chunks, f)
