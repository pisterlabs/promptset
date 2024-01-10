import re
from typing import Callable, Dict, List, Tuple
import langchain.docstore.document as docstore
import langchain.text_splitter as splitter
from loguru import logger
# local imports
from ingest.file_parser import FileParser


class IngestUtils:
    """
    Utils for ingesting different types of documents.
    This includes cutting text into chunks and cleaning text.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, file_no: int, text_splitter_method: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_no = file_no
        self.text_splitter_method = text_splitter_method

    def clean_text_to_docs(self, raw_pages, metadata) -> List[docstore.Document]:
        cleaning_functions: List = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines
        ]
        cleaned_text = self.clean_text(raw_pages, cleaning_functions)
        return self.text_to_docs(cleaned_text, metadata)

    def clean_text(self,
                   pages: List[Tuple[int, str]],
                   cleaning_functions: List[Callable[[str], str]]
                   ) -> List[Tuple[int, str]]:
        """
        Apply the cleaning functions to the text of each page.
        """
        logger.info("Cleaning text of each page")
        cleaned_pages = []
        for page_num, text in pages:
            # cnt = 0
            # print(f"Text before cleaning: {text}")
            for cleaning_function in cleaning_functions:
                # print(f"cleaning phase {cnt}")
                # cnt += 1
                text = cleaning_function(text)
                # print(text)
            cleaned_pages.append((page_num, text))
        return cleaned_pages

    def merge_hyphenated_words(self, text: str) -> str:
        """
        Merge words in the text that have been split with a hyphen.
        """
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def fix_newlines(self, text: str) -> str:
        """
        Replace single newline characters in the text with spaces.
        """
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def remove_multiple_newlines(self, text: str) -> str:
        """
        Reduce multiple newline characters in the text to a single newline.
        """
        return re.sub(r"\n{2,}", "\n", text)

    def get_splitter(self):
        """
        Get the text splitter object
        """
        if self.text_splitter_method == "NLTKTextSplitter":
            text_splitter = splitter.NLTKTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.text_splitter_method == "RecursiveCharacterTextSplitter":
            text_splitter = splitter.RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=self.chunk_overlap
            )
        return text_splitter

    def text_to_docs(self, text: List[Tuple[int, str]],
                     metadata: Dict[str, str]) -> List[docstore.Document]:
        """
        Split the text into chunks and return them as Documents.
        """
        doc_chunks: List[docstore.Document] = []

        chunk_no = 0
        for page_num, page in text:
            logger.info(f"Splitting page {page_num}")
            text_splitter = self.get_splitter()
            chunks = text_splitter.split_text(page)
            for i, chunk in enumerate(chunks):
                if self.file_no:
                    metadata_combined = {
                        "file_no": self.file_no,
                        "chunk_no": chunk_no,
                        "source": f"F{self.file_no}-{chunk_no}"
                    }
                else:
                    metadata_combined = {
                        "page_number": page_num,
                        "chunk": i,
                        "source": f"p{page_num}-{i}",
                        **metadata,
                    }
                doc = docstore.Document(
                    page_content=chunk,
                    metadata=metadata_combined
                )
                doc_chunks.append(doc)
                chunk_no += 1
        return doc_chunks
