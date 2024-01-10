from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from components.PDFExtractorEnum import PDFExtractorEnum
import logging

logger = logging.getLogger(__name__)

class PDFExtractor:
        
    @staticmethod
    def extract_docs_from_PDF(file, pdf_extractor_type):
        match pdf_extractor_type:
            case PDFExtractorEnum.PyPDFLoader: 
                # Split PDFs into 1 document per page
                loader =  PyPDFLoader(file_path=file)
                return loader.load()            
            case _:
                # Split PDFs into chunks of 1000 characters
                reader = PdfReader(file)
                raw_text = ''
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        raw_text += text

                text_splitter = RecursiveCharacterTextSplitter(        
                    chunk_size = 1000,
                    chunk_overlap  = 200,
                    length_function = len,
                )
                texts = text_splitter.split_text(raw_text)
                return text_splitter.create_documents(texts)        

