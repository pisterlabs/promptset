import os
import logging
import pickle
import ssl

import dill
import langchain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, GooglePalm
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain, AnalyzeDocumentChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import TextLoader, UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

class Vectorizer():
    llm = OpenAI(temperature=0.7, max_tokens=1024)
    embeddings = OpenAIEmbeddings()
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    
    def __init__(self, file_path: str):
        self.file_path = os.path.join(os.getcwd(), 'vectors', f'{file_path[:-4]}.pkl')
        
    def vector(self, split_docs: list, ) -> bool:
        self.logger.info('docs: %s', len(split_docs))
        # Using OpenAIEmbeddings models to provide further correlational data for our resulting vector for better semantic relationship identification
        vector_index = FAISS.from_documents(split_docs, self.embeddings)
        self.logger.info('Vector embedding created')
        
        # Exclude SSLContext from pickling
        dill._dill._reverse_typemap[type(ssl.create_default_context())] = None
        
        with open(self.file_path, 'wb') as f:
            dill.dump(vector_index, f)
            self.logger.info('Vector index saved')
        
        return True
    
    def load_index(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                vector_index = dill.load(f)
                self.logger.info('Vector index loaded')
            
            return vector_index
        else:
            self.logger.info('Vector index not found at the provided file path')
            return False
