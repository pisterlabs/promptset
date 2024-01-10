import os
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import utilities.token_helper as token_helper

from documents.document_loader import DOCUMENT_TYPES

def get_token_length_for_document(file_path: str) -> int:    
    '''This mostly gets a good representation of the token count in a document.'''
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_TYPES.get(file_extension)
    
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    
    document_text = "\n\n".join([d.page_content for d in loader.load()])# .load_and_split()
    
    return token_helper.num_tokens_from_string(document_text)