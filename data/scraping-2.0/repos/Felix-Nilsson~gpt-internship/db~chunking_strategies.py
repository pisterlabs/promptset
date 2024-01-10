from typing import IO, List
import json
import tiktoken
import chromadb
import re
import sys, os

from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.fileparser import pdf_to_plaintext

def num_tokens_from_string(string: str, encoding_name: str ="cl100k_base") -> int:
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class patient:
    def chunk_json(f : IO[str]) -> List[str]:
        """Splits json doc into chunks by keys, ~500 tokens"""
        json_obj = json.load(f)
        chunks = [json_obj[key] for key in json_obj.keys() if key not in ["prescription", "journal"]] 
        if "prescription" in json_obj.keys():
            chunks += json_obj["prescription"]
        if "journal" in json_obj.keys():
            chunks += json_obj["journal"]

        return chunks
    
    def chunk_ics(f : IO[str]) -> List[str]:
        """Function that chunks an .ics file. One chunk corresponds to one event, beggining with BEGIN:VEVENT and ending with END:VEVENT."""
        pattern = r"BEGIN:VEVENT(.*?)END:VEVENT"
        chunks = re.findall(pattern, f.read(), re.DOTALL)
        
        return chunks
    
    def add_to_collection(
            chunks : List[str], 
            collection : chromadb.api.models.Collection.Collection, 
            dir : str,
            filetype: str
        ):
        for i, chunk in enumerate(chunks):
            id = dir.split("/")[-1]
            collection.add(
                documents=[str(chunk)],
                metadatas=[
                    {
                        "patient":id, 
                        "type":filetype, 
                        "chunk_size": num_tokens_from_string(str(chunk)),
                        "chunk_index":i 
                    }
                ],
                ids=[f"{dir}_{i}_{filetype}"]
            )

class intranet:
    def chunk_pdf(dir : str) -> List[str]:
        pdf = pdf_to_plaintext(dir)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 200,
            chunk_overlap  = 20,
            length_function = num_tokens_from_string,
            add_start_index = True,
        )
        chunks = text_splitter.split_text(pdf)
        
        return chunks
    
    def add_to_collection(
            chunks : List[str],
            collection : chromadb.api.models.Collection.Collection,
            dir: str
        ):
        formatted_filename = dir.split("/")[-1]

        for i, chunk in enumerate(chunks):
                        
            collection.add(
                documents=[str(chunk)],
                metadatas=[
                    {
                        "doc":str(formatted_filename), 
                        "type":"pdf", 
                        "chunk_size": num_tokens_from_string(str(chunk)),
                        "chunk_index":i 
                    }
                ],
                ids=[f"{formatted_filename}_{i}"]
            )
