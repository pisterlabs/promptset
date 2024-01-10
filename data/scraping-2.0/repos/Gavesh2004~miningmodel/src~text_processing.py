from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessing:
    def __init__(self, pdf_files, chunk_size=740, chunk_overlap=20, separators=["\n"]):
        self.pdf_files = pdf_files
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    def process_text(self):
        for pdf in self.pdf_files:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            yield self.text_splitter.split_text(text)