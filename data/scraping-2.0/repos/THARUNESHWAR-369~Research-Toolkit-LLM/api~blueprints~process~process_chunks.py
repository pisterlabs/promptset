
from typing import Iterable, List, Any, cast
import json
from flask import jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter

from blueprints.dto.documents import Documents

class PROCESS_CHUNKS:
    """
    Class for processing chunks of data.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = ['\n\n', '\n', '.']
    
    @staticmethod
    def process_chunks(data: List) -> Any:
        """
        Process chunks of data.

        Args:
            data (List): List of data to be processed.

        Returns:
            Any: Processed data in JSON format.
        """

        text_splitter = RecursiveCharacterTextSplitter(
            separators=PROCESS_CHUNKS().separator,
            chunk_size=PROCESS_CHUNKS().chunk_size,
            chunk_overlap=PROCESS_CHUNKS().chunk_overlap
        )
        
        documents = [Documents().from_dict(doc) for doc in data]
        
        docs = text_splitter.split_documents(documents=documents)
        
        dict_data = [Documents(page_content=str(doc.page_content), source=doc.metadata['source']).to_dict() for doc in docs] # type: ignore
        
        return json.dumps(dict_data)
        
        