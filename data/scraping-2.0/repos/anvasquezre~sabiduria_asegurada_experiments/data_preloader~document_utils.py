from langchain.text_splitter import CharacterTextSplitter
import tiktoken
import config
from typing import Optional, List, Tuple, Union, Any,Dict
from pathlib import Path
from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document
import PyPDF2
import re
import os
from langchain.vectorstores import Qdrant
import text_preprocessing as tp
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import unstructured


def load_docs(sub_path:[str]) -> list[Document]:
    """ Load documents from a subpath to Document class for further use in Qdrant
    
    Args:
        sub_path (str]): subpath to load documents from (e.g. "raw_pdfs", "raw_chunks")

    Returns:
        list[Document]: List of Document objects
    """    
    load_path = str(Path(config.DATASET_ROOT_PATH) /  sub_path)
    loader = DirectoryLoader(load_path)
    docs = loader.load()
    return docs
    
def create_splitter() -> CharacterTextSplitter:
    """ Create a CharacterTextSplitter object based on config.py

    Returns:
        CharacterTextSplitter: CharacterTextSplitter 
    """    
    tiktoken_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    model_name=config.TIKTOKEN_EMBBEDINGS_MODEL,
    chunk_size=int(config.CHUNK_SIZE),
    chunk_overlap=int(config.CHUNK_OVERLAP),
    separator="[articulo]{0,1}[ARTICULO]{0,1}[ARTÍCULO]{0,1}[Artículo]{0,1}",
    keep_separator=True,
)
    return tiktoken_splitter

def get_chunks(docs: List[Document], 
            splitter: Optional[CharacterTextSplitter] | None = None) -> list[str]:
    """ Split doc into chunks

    Args:
        text (str): text to be splitted
        splitter (CharacterTextSplitter): CharacterTextSplitter object

    Returns:
        list: list of splitted text
    """
    if not splitter:
        splitter = create_splitter()
    
    chunks = splitter.split_documents(docs)
    return chunks

def get_policies_index(pdf_list:Optional[list[str]] | None = None) -> list[dict[str,int,int,str,int]]:
    """ Get policies index from pdf_list in directory, format: {"policy_number": str, "start_page": int, "end_page": int, "filepath": str, "num_pages": int}

    Args:
        pdf_list (Optional[list[str]], optional): List of strings with paths. Defaults to None.

    Returns:
        list[dict[str,int,int,str,int]]: _description_
    """    
    policies_list = []
    for pdf in pdf_list:
        pattern = r"([A-za-z]{3}\d{3,15})"
        filepath = str(Path(config.DATASET_ROOT_PATH) /  f"raw_pdfs/{pdf}")
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pattern_found = False
            start_page = None
            

            policies_metadata = {"policy_number": None, "start_page": None, "end_page": None, "filepath": None, "num_pages": None}
            num_pages = len(pdf_reader.pages)
            policies_num = 0
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                pattern_found = re.findall(pattern, page_text)
                
                
                if pattern_found:
                    if policies_num == 0:
                        policies_num += 1
                        policies_metadata = {"policy_number": pattern_found[0], "start_page": page_num, "end_page": None, "filepath": filepath, "num_pages": None}
                        policies_list.append(policies_metadata)
                    elif policies_num > 0:
                        policies_num += 1
                        end_page = page_num
                        policies_list[-1]["end_page"] = end_page-1
                        start_page = policies_list[-1]["start_page"]
                        num_pages = end_page - start_page
                        policies_list[-1]["num_pages"] = num_pages
                        policies_metadata = {"policy_number": pattern_found[0], "start_page": page_num, "end_page": None, "filepath": filepath, "num_pages": None}
                        policies_list.append(policies_metadata)
        policies_list[-1]["num_pages"] = num_pages
        policies_list[-1]["end_page"] = page_num
        
    return policies_list


def split_by_index(policies_list:list[dict[str,int,int,str,int]]) -> None:
    """ Split pdfs into new pdfs based on policies_list indexing

    Args:
        policies_list (list[dict[str,int,int,str,int]]): Dict with policy mapping
    Returns:
        None: None
    """    
    for doc in policies_list:
        pn, sp, ep, fp, np = doc.values()
        
        if not os.path.exists(f'{config.DATASET_ROOT_PATH}/raw_chunks'):
            os.makedirs(f'{config.DATASET_ROOT_PATH}/raw_chunks')
            
        with open(fp, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            output_pdf = PyPDF2.PdfWriter()
            output_file = f'{config.DATASET_ROOT_PATH}/raw_chunks/{pn}.pdf'
            if not os.path.exists(output_file):
                for i in range(sp, ep+1):
                    output_pdf.add_page(pdf_reader.pages[i])
                with open(f'{config.DATASET_ROOT_PATH}/raw_chunks/{pn}.pdf', 'wb') as output_file:
                        output_pdf.write(output_file)
            else:
                print(f"""File {pn} already exists.
                    check manually for inconsistencies in file {fp} with the following original raw doc""")
                
    return 

def divide_policies(pdf_list:Optional[list[str]] | None = None):
    """ Divide policies into policy chunks and save them in raw_chunks folder
    """
    if pdf_list:    
        policies_list = get_policies_index(pdf_list)
        split_by_index(policies_list)
        return print("Policies divided into subpolicies")
    else:
        return print("No pdfs provided")


def load_embeddings() -> Any:
    """ Load HuggingFaceEmbeddings

    Returns:
        Any: embeddings model
    """    
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBBEDINGS_MODEL)
    return embeddings

def load_from_docs_qdrant(docs:list[Document],
                        embeddings:Any , 
                        collection_name: Optional[str] = config.COLLECTION_CHUNKS,
                        ) -> None:
    """ Load documents into Qdrant

    Args:
        docs (list[Document]): List of Document objects
        collection_name (str): Name of collection in Qdrant
    """
    
    db_qdrant_chunks = Qdrant.from_documents(docs, 
                        embedding=embeddings,
                        collection_name=collection_name,
                        url = config.QDRANT_HOST,
                        prefer_grpc=True)
    
    return 

def load_from_texts_qdrant(docs:list[str],
                        embeddings:Any , 
                        metadata: List[dict[str, Any]] = None,
                        collection_name: Optional[str] = config.COLLECTION_SUMMARY) -> None:
    """ Load documents into Qdrant from texts

    Args:
        docs (list[str]): List of Document objects
        embeddings (Any): Embeddings function
        metadata (List[dict[str, Any]], optional): List of metadata. Defaults to None.
        collection_name (Optional[str], optional): Collection name. Defaults to config.COLLECTION_CHUNKS.

    Returns:
        None: None
    """
    
    db_qdrant_chunks = Qdrant.from_texts(
                        texts=docs,
                        embedding=embeddings,
                        collection_name=collection_name,
                        metadatas=metadata,
                        url = config.QDRANT_HOST,
                        prefer_grpc=True)
    
    return 

def map_docs_metadata(docs:list[Document]
                      ) -> List[dict[str, Any]]:
    """ Map metadata from Document objects

    Args:
        docs (list[Document]): List of Document objects

    Returns:
        List[dict[str, Any]]: List of metadata
    """
    MODEL_16K = "gpt-3.5-turbo-16k"
    encoder = tiktoken.encoding_for_model(MODEL_16K)
    text_dict = {} # Key: source, Value: dict with num_tokens, num_articles, text, title
    pattern = r"(\narticulo .*.*\s*\d+\d*).*:{0,1}\.*\s*(\s*[^\n]*)((?=[\s*]*))"

    for doc in docs:
        content = doc.page_content
        text_w_char = tp.normalize_corpus([content], special_char_removal=False, stopword_removal=False,remove_digits=False)
        num_tokens = len(encoder.encode(content))
        articles = len(re.findall(pattern, text_w_char[0]))
        title = text_w_char[0].split(r"articulo")[0]
        text_dict[doc.metadata["source"]] = {"num_tokens": num_tokens , "num_articles": articles, "text": text_w_char, "title": title}

    return text_dict