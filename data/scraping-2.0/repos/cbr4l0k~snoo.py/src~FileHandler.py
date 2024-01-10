import json
import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import (RecursiveCharacterTextSplitter, Language)
from langchain.schema.document import Document

# some important enviroment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
PROJECTS_PATH = os.getenv("PROJECTS_PATH")
OUTPUTS_PATH = os.getenv("OUTPUTS_PATH")


class FileHandler:

    """
        This class is supposed to handle the files, individually.
        It will be in charge of reading the files, chunking them, and retrieving the chunks.


        Attributes:
        ----------
        - None 
        
        Methods:
        ----------

        - chunk_document: List[Document]
            This function chunks a given block of code taking into account the semantic categories given in a language
            by considering it's syntax, from it recursively tries to divide each chunk into one or many of the desired
            chunk size, this does not guarantee that they all have the same size, but they should be close.
            Also considers the chunk overlap, which allows to have a bit of the previous information available.
        
        - from_filename_to_lang: Language
            This function is supposed to take the filename and infer the language by taking the last part of the name
            and then return the Language it corresponds to if any (in the dict of supported LangChain languages)
            If it does not find it, it will return None.
    """

    def __init__(self) -> None:
        """
            The constructor for the FileHandler class.
        """
        pass 


    @staticmethod
    def from_filename_to_lang(filename: str):
        """
            This function is supposed to take the filename and infer the language by taking the last part of the name
            and then return the Language it corresponds to if any (in the dict of supported LangChain languages)
            If it does not find it, it will return None

            Args:
            ----------
            filename: str
                The filename to infer the language from
            
            Returns:
            ----------
            Language
                The language that corresponds to the filename, if any.
        """
        return Language.PYTHON

    @staticmethod
    def chunk_document(filename_full_path: str, code: str, chunk_size: int, chunk_overlap: int = 0) -> List[Document]:
        """
            This function chunks a given block of code taking into account the semantic categories given in a language
            by considering it's syntax, from it recursively tries to divide each chunk into one or many of the desired
            chunk size, this does not guarantee that they all have the same size, but they should be close.
            Also considers the chunk overlap, which allows to have a bit of the previous information available.

            Args:
            ----------
            filename_full_path: str
                The full path to the file, including the filename and extension.
            code: str
                The code to chunk.
            chunk_size: int
                The size of the chunks to create.
            chunk_overlap: int
                The overlap between chunks.
            
            Returns:
            ----------
            List[Document]
                The list of documents created from the chunking.
        """

        filename = filename_full_path.split("/")[-1]
        lang = FileHandler.from_filename_to_lang(filename)
        python_splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=chunk_size,
                                                                       chunk_overlap=chunk_overlap,
                                                                       language=lang, )
        docs = python_splitter.create_documents([code])
        return docs