import os
import fnmatch
from typing import Iterator, List

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MarkdownLoader():

    def __init__(self, markdown_path: str):
        self.markdown_path = markdown_path

    # find all .md files recursively and save the path into metadata
    def load_batch(self) -> Iterator[Document]:
        # walk through all files
        for dirpath, dirs, files in os.walk(self.markdown_path): 
            for filename in fnmatch.filter(files, '*.md'):
                md_filepath = os.path.join(dirpath, filename)
                
                # split one file into multiple files based on headers
                splitted_md_files = self.split_markdown_file(md_filepath)

                for file in splitted_md_files:
                    # add relative location and filename of data to metadata
                    file.metadata["dirpath"] = dirpath
                    file.metadata["filename"] = filename
                    yield file # lazy loading of files
    
    def split_markdown_file(self, md_filepath: str) -> List[Document]:
        
        md_file = open(md_filepath, "r", encoding="utf-8").read()
        
        # logical split with headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(md_file)
        
        # text should not exceed 256 words
        def length_function(text: str) -> int:
            return len(text.split())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1024,
            chunk_overlap = 100,
            length_function = length_function
        )
        splits = text_splitter.split_documents(md_header_splits)

        return splits