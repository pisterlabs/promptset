from fastapi import UploadFile
from PyPDF2 import PdfReader
import io
from json import loads
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from models.parametersModel import Parameters
async def construct_pdf(file: UploadFile, strict: bool = False, password: str = None) -> PdfReader:
    """Creates a ``PdfReader`` object from a ``UploadFile`` object.
    
    Paraemters:
    
    - ``file``: a FastAPI ``UploadFile`` object or any wrapper around the python´s file API representing a pdf file.
    - ``strict``: a boolean indicating if the pdf file should be read in strict mode.
    - ``password``: a string representing the password to be used to read the pdf file.
    Returns:
    
    - ``pdf_file``: a ``PdfReader`` object representing the pdf file.
    
    Raises:
    
    ``Exception`` if an error occurs while creating the ``PdfReader`` object.
    """
    try:
        raw_file_content = await file.read()
        bytes_stream = io.BytesIO(raw_file_content)

        pdf_file = PdfReader(bytes_stream, strict=strict, password=password)

        return pdf_file
    except Exception as e:
        raise e.add_notes("Error while creating pdf file")


def parse_json(string: str) -> dict:
    """Parses a stringified version of a json object into a python´s dictionary.
    
    Parameters:
    
    - ``string``: a stringified version of a json object.
    
    Returns
    
    a python´s dictionary representing the json object.
    
    Raises:
    
    ``Exception`` if an error occurs while parsing the json string.
    """
    try:
        return loads(string)
    except Exception as e:
        raise e.add_notes("Error while parsing json string")





def validate_schema(json_data: dict) -> Parameters:
    """validates a dictionary against the ``Parameters`` Pydantic model.
    
    Parameters:
    
    - ``json_data``: a python dictionary containing the parameters.
    
    Returns
    
    a Pydantic ``Parameter``model containing the LLMs parameters
    
    Raises:
    
    ``Exception`` if an error occurs while validating the dictionary.
    """
    try:
        parsed_parameters = Parameters(**json_data)
        return parsed_parameters
    except Exception as e:
        raise e.add_notes("Error while validating parameters")



def extract_text(pdf_file: PdfReader,start:int=0,finish:int=-1) -> str:
    """Extracts all the text from a ``PdfReader`` object.
    
    Parameters:
    
    - ``pdf_file``: a PdfReader object representing a pdf file.
    
    - ``start``: the page number to start extracting text from.
    
    - ``finish``: the page number to stop extracting text from.
    
    Return:
    
    ``full_text``: a string containing all the text extracted from the pdf file.
    
    Raises:
    
    ``IndexError`` if the ``start`` or ``finish`` parameters are out of range related to the pdf pages.
    """
    try:
        pages = pdf_file.pages[0:] if finish == -1 else pdf_file.pages[start:finish]
        full_text = "\n\n".join([page.extract_text() for page in pages])
        full_text = full_text.replace('\t', ' ')
        return full_text
    except IndexError as e:
        raise e.add_notes("Error while extracting text from pdf file - index out of range")


def get_documents(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """creates langchain´s Document objects from a string of text corpus using a recursive character text splitter
    
    Parameters:
    
    - ``text`` a string representing a text corpus.
    
    - ``separators`` a list of strings patterns representing the separators to be used by the text splitter.
    
    - ``chunk_size`` the size of the chunks to be created by the text splitter.
    
    - ``chunk_overlap`` the overlap between chunks to be created by the text splitter.
    
    Returns:
    
   ``documents``: A list of langchain´s Document objects containing the chunks created by the text splitter.
    
    Raises:
    
    ``Exception`` if an error occurs while creating the documents.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.create_documents([text])
        return documents
    except Exception as e:
        raise e.add_notes("Error while creating documents from text corpus")
