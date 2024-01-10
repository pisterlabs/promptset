from typing import Dict, List, Tuple
from loguru import logger
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from pypdf import PdfReader
# local imports
import utils as ut


class FileParser:
    """A parser for extracting text from .txt documents."""

    def __init__(self, chunk_size: int, chunk_overlap: int, file_no: int, text_splitter_method: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_no = file_no
        self.text_splitter_method = text_splitter_method

    def parse_file(self, file_path: str):
        if file_path.endswith(".pdf"):
            raw_pages, metadata = self.parse_pdf(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            raw_pages, metadata = self.parse_txt(file_path)
        elif file_path.endswith(".html"):
            raw_pages, metadata = self.parse_html(file_path)
        elif file_path.endswith(".docx"):
            raw_pages, metadata = self.parse_word(file_path)
        return raw_pages, metadata
    
    def get_metadata(self, file_path: str, metadata_text: str):
        return {"title": ut.getattr_or_default(obj=metadata_text, attr='title', default='').strip(),
                "author": ut.getattr_or_default(obj=metadata_text, attr='author', default='').strip(),
                "filename": file_path.split('\\')[-1]
                }

    def parse_html(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the html file"""
        # load text and extract raw page 
        logger.info("Extracting pages")
        loader = BSHTMLLoader(file_path, open_encoding='utf-8')
        data = loader.load()
        raw_text = data[0].page_content.replace('\n', '')
        pages = [(1, raw_text)] # html files do not have multiple pages

        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = data[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        return pages, metadata

    def parse_pdf(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the PDF file"""
        metadata = self.extract_metadata_from_pdf(file_path)
        pages = self.extract_pages_from_pdf(file_path)
        return pages, metadata

    def extract_metadata_from_pdf(self, file_path: str) -> Dict[str, str]:
        """Extract and return the metadata from the PDF file"""
        logger.info("Extracting metadata")

        with open(file_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            metadata_text = reader.metadata
            logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
            metadata = self.get_metadata(file_path, metadata_text)
            return metadata

    def extract_pages_from_pdf(self, file_path: str) -> List[Tuple[int, str]]:
        """Extract and return the text of each page from the PDF file"""
        logger.info("Extracting pages")
        with open(file_path, "rb") as pdf:
            reader = PdfReader(pdf)
            return [(i + 1, p.extract_text()) for i, p in enumerate(reader.pages) if p.extract_text().strip()]

    def parse_txt(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the text file"""
        # load text and extract raw page 
        logger.info("Extracting pages")
        loader = TextLoader(file_path)
        text = loader.load()
        raw_text = text[0].page_content
        pages = [(1, raw_text)] # txt files do not have multiple pages

        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = text[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        return pages, metadata
    
    def parse_word(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """Extract and return the pages and metadata from the word document."""
        # load text and extract raw page 
        logger.info("Extracting pages")
        loader = UnstructuredWordDocumentLoader(file_path)
        text = loader.load()
        raw_text = text[0].page_content
        pages = [(1, raw_text)] # currently not able to extract pages yet!

        # extract metadata
        logger.info("Extracting metadata")
        metadata_text = text[0].metadata
        logger.info(f"{getattr(metadata_text, 'title', 'no title')}")
        metadata = self.get_metadata(file_path, metadata_text)
        return pages, metadata
