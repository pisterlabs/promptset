
from langchain.vectorstores import Pinecone
from typing import List, Any
import os

from blueprints.dto.documents import Documents

class DB_PINECONE:
    
    @staticmethod
    def from_documents(docs: List[Documents], 
                       openai_embeddings : Any, 
                       index_name : str = os.environ['PINECONE_INDEX_NAME']) -> Any:
        
        return Pinecone.from_documents(docs, openai_embeddings, index_name=index_name)
