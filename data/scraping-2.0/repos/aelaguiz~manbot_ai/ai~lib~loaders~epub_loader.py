import sys
import os
from typing import List
from bs4 import BeautifulSoup
import json
from ebooklib import epub
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
import logging
import html2text

h = html2text.HTML2Text()
h.ignore_links = False
import ebooklib

class EPubLoader(UnstructuredFileLoader):

    def load(self) -> List[Document]:
        # Get base filename
        filename = os.path.basename(self.file_path)
        docs = []

        book = epub.read_epub(self.file_path)
        page_content = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                html_content = str(soup)
                markdown_content = h.handle(html_content)
                page_content += "\n" + markdown_content

        doc = Document(page_content=page_content, metadata={
            "title": book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else 'Unknown Title',
            "author": book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else 'Unknown Author',
            'type': 'book',
            "filename": filename
        })

        docs.append(doc)

        
        return docs