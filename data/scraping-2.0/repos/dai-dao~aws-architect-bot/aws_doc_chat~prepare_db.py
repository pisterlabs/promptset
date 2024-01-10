import os
import openai
from dotenv import load_dotenv
load_dotenv()
from langchain.docstore.document import Document
openai.api_key = os.environ["OPENAI_API_KEY"]
from vector_db import VectorDB, QdrantVectorDB
from constants import *
import json
from typing import List


# Load the document, split it into chunks, embed each chunk and load it into the vector store.
def build_section_titles_index():
    with open(CHAINED_TITLES_CONTENT_OUTPUT, "r") as f:
        chained_titles_content = json.load(f)
        documents = []
        for content_dict in chained_titles_content:
            doc = Document(page_content = content_dict['chained_title'])
            documents.append(doc)

    vector_db = QdrantVectorDB(SECTION_TITLES_INDEX_NAME)
    vector_db.build_index(documents)
    
    
# TODO: AN Optimization
# chunk each section content by overlapping chunks
# Then put metadata, like section titles, and chunk order into the DB

# NOTE: NOT storing urls in metadata
def build_section_content_index():
    
    def format_urls(urls) -> List[str]:
        out = []
        for url, text in urls:
            out.append(f"[{text}]({url})")
        return out
        
    
    with open(CHAINED_TITLES_CONTENT_OUTPUT, "r") as f:
        chained_titles_content = json.load(f)
        documents = []
        for content_dict in chained_titles_content:
            doc = Document(page_content = content_dict['content'], 
                           metadata = {"chained_title" : content_dict['chained_title']})
            documents.append(doc)

        vector_db = VectorDB(CHAINED_TITLES_SECTION_CONTENT_INDEX_NAME)
        vector_db.build_index(documents)
    

# When query, if i get the metadata, i can get the previous chunk as well to complete the section
# build_section_content_index()

# build_section_titles_index()
