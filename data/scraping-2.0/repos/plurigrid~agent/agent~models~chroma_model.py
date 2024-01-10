from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
import os

class ChromaModel: 
    def __init__(self, document_dir):
        self.document_dir = document_dir
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma()
        
        # Create an instance of the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        # Iterate over all files in the directory
        for file_name in os.listdir(document_dir):
            # Open the file and read its contents
            with open(os.path.join(document_dir, file_name)) as f:
                file_contents = f.read()
                
                # Split the file contents into chunks
                texts = text_splitter.split_text(file_contents)
                docsearch.add_texts(texts)

