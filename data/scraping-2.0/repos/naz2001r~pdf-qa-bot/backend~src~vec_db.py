import pickle
import logging
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


# Configure logging
logging.basicConfig(level=logging.INFO)


class VectorizeDB:
    """
    A class for vectorizing datasets.
    """

    def __init__(self, openai_key: str) -> None:
        """
        Initialize a VectorizeDB object.

        Args:
            openai_key (str): OpenAI API key (default is an empty string).
        """

        assert isinstance(openai_key, str), "openai_key must be a string"
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        self.__db = None
        self.__retriever = None

    def vectorize(self, pages: list, extend: bool = False) -> None:
        """
        Vectorize a list of pages from pdf files and create a vector database.

        Args:
            pages (list): List of pages to vectorize.
            extend (bool): If True, extend the existing database with new pages.
                           If False, create a new database (default is False).
        """

        assert isinstance(pages, list), "pages must be a list"
        assert isinstance(extend, bool), "extend must be a boolean"

        logging.info("Vectorizing pages...")
        if self.__db is not None and extend:
            db_new = FAISS.from_documents(pages, self.embeddings)
            self.__db = self.__db.merge_from(db_new)
        else:
            self.__db = FAISS.from_documents(pages, self.embeddings)

    @property
    def retriever(self) -> object:
        """
        Get the current retriever object.

        Returns:
            object: Retriever object.
        """

        return self.__retriever

    @retriever.setter
    def retriever(self, k: int = 5) -> None:
        """
        Set the retriever object with the specified number of query output.

        Args:
            k (int): Number of query output (default is 5).
        """

        if not isinstance(k, int):
            raise TypeError(f"Type {type(k)} is not supported for the number of query output `k`")

        logging.info(f"Setting retriever with k={k}...")
        self.__retriever = self.__db.as_retriever(search_kwargs={"k": k})

    def query(self, text: str) -> list:
        """
        Query the vector database to retrieve relevant documents.

        Args:
            text (str): Text to query.

        Returns:
            list: List of relevant documents.

        Raises:
            TypeError: If the retriever object is not set.
        """

        assert isinstance(text, str), "text must be a string"

        if self.retriever:
            logging.info(f"Querying with text: {text}")
            return self.retriever.get_relevant_documents(text)
        raise TypeError('Please set retriever before calling it.')

    @classmethod
    def load_db(cls, file_name: str) -> object:
        """
        Load a VectorizeDB object from a pickle file.

        Args:
            file_name (str): Name of the pickle file.

        Returns:
            object: Loaded VectorizeDB object.
        """

        assert isinstance(file_name, str), "file_name must be a string"
        logging.info(f"Loading VectorizeDB from file: {file_name}")
        return pickle.load(open(file_name, 'rb'))

    def dump_db(self, file_name: str) -> None:
        """
        Dump the VectorizeDB object to a pickle file.

        Args:
            file_name (str): Name of the pickle file.
        """

        assert isinstance(file_name, str), "file_name must be a string"
        logging.info(f"Dumping VectorizeDB to file: {file_name}")
        pickle.dump(self, open(file_name, 'wb'))
