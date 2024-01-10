import pinecone
from utils.chunked_upsert import chunked_upsert
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import os
from hashlib import md5
from langchain.document_loaders import PyPDFLoader
from utils.truncateStringByBytes import truncate_string_by_bytes
import asyncio
from utils.embeddings import get_embeddings
from flask import jsonify
import logging 
from enum import Enum

logging.basicConfig(
    level=logging.DEBUG,  # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

class SplittingOptions(Enum):
    SENTENCE = 'sentence'
    CHAR = 'character'
    RECUR = 'recursive'

@dataclass
class SeedOptions:
    chunk_size: int = 1500
    chunk_overlap: int = 50
    method: SplittingOptions = SplittingOptions.CHAR

async def parse_pdf(pdf_file):
    logging.debug(f"Type of pdf_file: {type(pdf_file)}")
    try:
        pdf_path = os.path.join('user-uploads', pdf_file)

        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        doc_strings = [{"content": page.page_content, "metadata": {"page_number": page.metadata['page'], "title": pdf_file}} for page in pages]
        return doc_strings
    except Exception as e:
        raise e

async def upload_and_generate_embedding(file, index_name: str, options: SeedOptions = SeedOptions()):
    try:
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
        logging.debug("Initialized Pinecone")
        
        parsed_pdf = await parse_pdf(file)
        logging.debug("Parsed PDF")
        
        index_list = pinecone.list_indexes()
        logging.debug("List of indexes: %s", index_list)
        
        if index_name not in index_list:
            pinecone.create_index(name=index_name, dimension=1536)
            logging.debug("Created index: %s", index_name)
        
        index = pinecone.Index(index_name)
        logging.debug("Initialized Pinecone Index")
            
            # Choose the splitter based on the method
        if SeedOptions.method == SplittingOptions.RECUR:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap)
        elif SeedOptions.method == SplittingOptions.CHAR:  # Default to character splitter
            text_splitter = CharacterTextSplitter(chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap)
        elif SeedOptions.method == SplittingOptions.SENTENCE:
            text_splitter = SpacyTextSplitter(chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap)
        chunked_pdf = await asyncio.gather(*[chunk_pdf(x, text_splitter) for x in parsed_pdf])
        chunked_pdf = [item for sublist in chunked_pdf for item in sublist]
        logging.debug("Chunked PDF and obtained vectors")
        
        vectors = await asyncio.gather(*[embed_chunks(chunk) for chunk in chunked_pdf])
        logging.debug("Embedded chunks")
        
        await chunked_upsert(index=index, vectors=vectors)
        logging.debug("Upserted vectors into index")
        
        return {"success": True, "message": "Successfully uploaded file(s)", "filename": file}
    except Exception as e:
        # Log the error here
        logging.debug("Error in upload_and_generate_embedding: %s", str(e))
        return {"success": False, "message": str(e)}

async def embed_chunks(doc):
    try:
        embedding = get_embeddings(doc["page_content"])
        hashed = md5(doc["page_content"].encode()).hexdigest()
        hashed_doc_id = md5(doc["metadata"]["title"].encode()).hexdigest()
        return {
            "id": hashed,
            "values": embedding,
            "metadata": {
                "chunk": doc["page_content"],
                "hashed": doc["metadata"]["hashed"],
                "page_number": doc["metadata"]["page_number"],
                "doc_id": hashed_doc_id
            }
        }
    except Exception as e:
        raise e
        
async def chunk_pdf(page, splitter):
    docs = splitter.create_documents([page["content"]])
    proc_docs = [{"page_content": doc.page_content, "metadata": {**doc.metadata, **page["metadata"], "hashed": md5(doc.page_content.encode()).hexdigest()}} for doc in docs]
    return proc_docs

    
