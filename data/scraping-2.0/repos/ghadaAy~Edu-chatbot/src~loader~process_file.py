import logging
from settings import get_settings
from typing import List, Union
from langchain.schema.document import Document
from src.loader.base import BaseProcessor
from src.loader.static import LOADER_MAPPING
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust the logging level as needed

app_settings = get_settings()
pytesseract.pytesseract.tesseract_cmd = app_settings.PATH_TESSERACT  # your path may be different

class ProcessFile(BaseProcessor):
    
    def load_single_document(self, source: str) -> Union[Document, None]:
        """Load a single document.

        Args:
            source (str): Path to the file.

        Returns:
            Document or None
        """
        try:
            ext = "." + source.rsplit(".", 1)[-1]
            ext = ext.lower()
            if ext not in LOADER_MAPPING:
                logging.error(f"{source} cannot be indexed")
                return None
            else:
                loader_class, loader_args = LOADER_MAPPING[ext]
                try:
                    loader = loader_class(source, **loader_args)
                    return loader.load()
                except Exception as e:
                    logging.error(f"Error loading document using {loader_class}: {e}")
                    return None
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            return None
        

    def process_document(self, source: str) -> Union[List[Document], None]:
        """Loads a document and recursively splits it to create docs.

        Args:
            source (str): Path to the file.

        Returns:
            list of Documents or None
        """
        logging.info(f"Processing file {source}")
        docs = self.load_single_document(source)
        if docs is not None:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            try:
                docs = text_splitter.split_documents(docs) # type: ignore
                logging.info("Processing successful")
                return docs
            except Exception as e:
                logging.error(f"Error splitting document: {e}")
                return None
