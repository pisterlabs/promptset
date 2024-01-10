import io

from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import streamlit as st

class DocumentLoader:
    """
    This class provides methods to load documents from PDF files.
    """
    def __init__(self,file_name) -> None:
        self.file_name = file_name


    def _read_pages(self,pdf_reader):
        """
        Reads the text from all pages of a PDF file.
        """

        pdf_title = self.file_name

        doc_data_list = []

        for page in pdf_reader.pages:

            doc_data = {
                'page_number': pdf_reader.get_page_number(page),
                'page_content': page.extract_text(),
                'pdf_title': pdf_title,
            }

            doc_data_list.append(doc_data)


        return doc_data_list

    def read_pdf_from_bytes(self,bytes_data):
        """
        Reads a PDF file from bytes data.
        """

        with io.BytesIO(bytes_data) as base64_pdf:
            reader = PdfReader(base64_pdf)
            doc = self._read_pages(reader)

        return doc

    @staticmethod
    def read_pdf_from_path(filepath, pdf_password=None):
        """
        Reads a PDF file from a file path.
        """

        reader = PyPDFLoader(filepath, password=pdf_password)

        return reader.load()


class StylesLoader:
    """
    This class provides methods to load CSS styles from a file.
    """

    @staticmethod
    def load(css_file_path):
        """
        Loads CSS styles from a file.
        """
        with open(css_file_path, 'r', encoding='utf-8') as f:
            css = f.read()
        return f'<style>\n\n{css}\n</style>'


class HtmlLoader:
    """
    This class provides methods to load HTML from a file.
    """

    @staticmethod
    def load(html_file_path):
        """
        Loads HTML from a file.
        """
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return html
