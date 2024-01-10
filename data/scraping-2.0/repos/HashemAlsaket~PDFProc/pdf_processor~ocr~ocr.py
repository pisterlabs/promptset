from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from typing import List


def process_pdf(pdf_file_path: str) -> List[str]:
    f"""
    Basic OCR text from PDF.
    """
    # Read text from PDF file
    reader = PdfReader(pdf_file_path)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    # Prepare text for LLM 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=5050,
        chunk_overlap=200,
        length_function=len
    )

    texts = text_splitter.split_text(raw_text)
    return texts