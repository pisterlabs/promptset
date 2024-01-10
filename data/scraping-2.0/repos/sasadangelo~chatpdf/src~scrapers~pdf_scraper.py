import PyPDF2
from langchain.text_splitter import TokenTextSplitter
from src.scrapers.scraper import Scraper

class PDFScraper(Scraper):
    def __init__(self, pdf_file_path):
        self.pdf_file_path = pdf_file_path

    def scrape(self):
        text = ""
        
        with open(self.pdf_file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for _, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text
        text = "\n".join(line for line in text.splitlines() if line.strip())

        text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
        return text_splitter.split_text(text)