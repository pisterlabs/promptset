from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.extractor.text_extractor import TextExtractor
from typing import Sequence

class PdfTextExtractor(TextExtractor):
    """
    A singleton class to extract text from PDF and split it into smaller chunks.

    Attributes:
        split_chunk_size (int): Maximum size of each split text.
        split_chunk_overlap (int): Overlap size between adjacent split texts.
        split_separator (str): Separator used for splitting texts.
        splitter (RecursiveCharacterTextSplitter): Text splitter.
    """
    
    def __init__(self) -> None:
        self.split_chunk_size = 512
        self.split_chunk_overlap = 64
        self.split_separator = "\n"
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.split_separator,
            chunk_size=self.split_chunk_size,
            chunk_overlap=self.split_chunk_overlap,
            length_function=len,
        )

    def extract_texts(self, path: str) -> Sequence[str]:
        """
        Extract text from a PDF file and split it into smaller chunks.

        Args:
            path (str): The path to the PDF file.

        Returns:
            Sequence[str]: A sequence of texts.
        """
        text = ''
        with open(path, 'rb') as file:
            pdf = PdfReader(file)
            for _, page in enumerate(pdf.pages):
                text += page.extract_text()
        
        # Split text into smaller chunks
        texts = self.splitter.split_text(text)
        return texts
