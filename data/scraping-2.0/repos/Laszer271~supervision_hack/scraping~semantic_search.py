import logging
from pathlib import Path
from typing import List, Tuple
import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document


LOGGER = logging.getLogger(__name__)


class SemanticSearch():
    """Class containing modules for the semantic search.
    """

    model_name: str
    model: HuggingFaceEmbeddings

    def __init__(self,
                 model_name: str = "sdadas/st-polish-paraphrase-from-distilroberta",
                 **kwargs
        ) -> None:
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(model_name=self.model_name, **kwargs)

        self.vectorstore = None

    # def vectorize_doc(self, doc: Path, vectordb_dir: Path) -> None:
    #     """Transform a doc containing all the information into a VectorDB.

    #     Args:
    #         doc (Path): File path containing the information. doc is a .txt file with /n/n/n separator.
    #         vectordb_path (Path, optional): _description_. Defaults to config.VECTORDB_PATH.
    #     """
    #     if doc.exists():
    #         with open(doc, "r") as f:
    #             text = f.read()
    #         texts = text.split("\n\n\n")
    #         LOGGER.info(f'Number of chunks: {len(texts)}')
    #         if self.vectorstore is None:
    #             # self.vectorstore = Chroma.from_texts(
    #             #     texts=texts, 
    #             #     embedding=self.model,
    #             #     #persist_directory=str(vectordb_dir) # Need to be a string
    #             # )
    #             self.vectorstore = FAISS.from_texts(
    #                 texts=texts, 
    #                 embedding=self.model,
    #             )
    #         else:
    #             self.vectorstore.add_texts(
    #                 texts=texts, 
    #                 embedding=self.model,
    #             )
    #     else:
    #         raise FileNotFoundError(f"{doc} does not exist.")
        
    def vectorize_text(self, strings: str) -> None:
        """Transform a doc containing all the information into a VectorDB.

        Args:
            doc (str): string containing the information.
            vectordb_path (Path, optional): _description_. Defaults to config.VECTORDB_PATH.
        """
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                texts=strings, 
                embedding=self.model,
            )
        else:
            self.vectorstore.add_texts(
                texts=strings, 
                embedding=self.model,
            )
        

    def search(self, query: str, vectordb_dir: str = str('../data/vectordb'),
            k: int = 1) -> List[Tuple[Document, float]]:
        """From a query, find the elements corresponding based on personal information stored in vectordb.
        Euclidian distance is used to find the closest vectors.

        Args:
        query (str): Question asked by the user.
        vectordb_dir (str, optional): Path to the vectordb. Defaults to config.VECTORDB_DIR.

        Returns:
        List[Tuple[Document, float]]: Elements corresponding to the query based on semantic search, associated
        with their respective score.
        """
        # timestamp = time.time()
        # vectordb = Chroma(persist_directory=vectordb_dir, embedding_function=self.model)
        results = self.vectorstore.similarity_search_with_score(query=query, k=k)
        # LOGGER.info(f"It took {time.time() - timestamp} to search elements with semantic search.")
        return results
    
    # def __del__(self):
    #     if self.vectorstore is not None:
    #         del self.vectorstore

