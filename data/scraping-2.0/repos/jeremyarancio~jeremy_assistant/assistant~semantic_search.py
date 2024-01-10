import logging
from pathlib import Path
from typing import List, Tuple
import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from assistant import config


LOGGER = logging.getLogger(__name__)


class SemanticSearch():
    """Class containing modules for the semantic search.
    """

    model_name: str
    model: HuggingFaceEmbeddings

    def __init__(self,
                 model_name: str = config.sbert_model_name,
                 **kwargs
        ) -> None:
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(model_name=self.model_name, **kwargs)

    def vectorize_doc(self, doc: Path, vectordb_dir: Path = config.VECTORDB_DIR) -> None:
        """Transform a doc containing all the information into a VectorDB.

        Args:
            doc (Path): File path containing the information. doc is a .txt file with /n/n/n separator.
            vectordb_path (Path, optional): _description_. Defaults to config.VECTORDB_PATH.
        """
        if doc.exists():
            with open(doc, "r") as f:
                text = f.read()
            texts = text.split(config.separator)
            LOGGER.info(f'Number of chunks: {len(texts)}')
            Chroma.from_texts(texts=texts, 
                              embedding=self.model, 
                              persist_directory=str(vectordb_dir) # Need to be a string
            )
            LOGGER.info(f"VectorDB correctly created at {vectordb_dir}")
        else:
            raise FileNotFoundError(f"{doc} does not exist.")
        
    def search(self, query: str, vectordb_dir: str = str(config.VECTORDB_DIR),
               k: int = config.k) -> List[Document]:
        """From a query, find the elements corresponding based on personal information stored in vectordb.
        Euclidian distance is used to find the closest vectors.

        Args:
            query (str): Question asked by the user.
            vectordb_dir (str, optional): Path to the vectordb. Defaults to config.VECTORDB_DIR.

        Returns:
            List[Tuple[Document, float]]: Elements corresponding to the query based on semantic search, associated
        with their respective score.
        """
        LOGGER.info("Enter vector search module.")
        timestamp = time.time()
        vectordb = Chroma(persist_directory=vectordb_dir, embedding_function=self.model)
        results = vectordb.similarity_search(query=query, k=k)
        LOGGER.info("Exit vector search module.")
        LOGGER.info(f"It took {time.time() - timestamp} to search elements with semantic search.")
        return results
