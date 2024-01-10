"""This module implements the embedding search of a pdf file"""
from typing import List
from paperplumber.logger import get_logger
from paperplumber.parsing.llmreader import OpenAIReader
from paperplumber.parsing.pdf_parser import PDFParser

logger = get_logger(__name__)


class FileScanner(PDFParser):
    """A class used to scan a PDF file for data using
    the OpenAIReader functionality."""

    def __init__(self, pdf_path: str):
        super(PDFParser, self).__init__(pdf_path)

    @classmethod
    def from_pages(cls, pages: List):
        """Creates a FileScanner object from a list of pages.

        Args:
            pages (List): A list of pages to be scanned.

        Returns:
            FileScanner: A FileScanner object with the specified pages."""

        scanner = cls.__new__(cls)
        scanner._pages = pages
        return scanner

    def scan(self, target: str) -> List[str]:
        """Scans the pages of a document for a specified target using the OpenAIReader.

        This function scans each page of the document and retrieves values related to
        the target, discarding any 'NA' values. If multiple unique values are found for
        the target, a warning is logged.

        Args:
            target (str): The target to be scanned within the document pages.

        Returns:
            List[str]: A list of unique values found for the target in the document pages,
                    excluding 'NA'. If no value is found, returns an empty list.

        Raises:
            Warning: If more than one unique value is found for the target."""

        reader = OpenAIReader(target)
        values = [reader.read(page.page_content) for page in self._pages]

        # Remove NAs
        clean_values = {value for value in values if value != "NA"}

        # Warn if multiple values are found
        if len(clean_values) > 1:
            logger.warning("Found multiple values for % as the target.", target)

        return list(clean_values)
