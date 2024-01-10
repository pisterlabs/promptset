# To handle the issue https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/5
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import Dict, List
import uuid
import math
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


class DocumentSource:
    def __init__(self,
                 openai_api_key: str):
        """
        Initializes with a dictionary of papers and an OpenAI API key.

        Args:
            papers (list[Document]): A list containing documents.
            openai_api_key (str): The OpenAI API key for text embeddings.

        Vector store elements are structured as follows:
        [
            'embedding': page_info,
            '[122143,123213,346346,34325234]': {
                page_content: 'LLM XXX',
                metadata: {
                    source: 'title ',
                    page: int,
                },
            },
        ]
        """
        self.num_docs_ = 0
        # Get embedding from OpenAI.
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # UUID4: Generates a random UUID in UUID class type.
        db_uuid = str(uuid.uuid4())
        # Compute embeddings for each chunk and store them in the database. Each with a unique id to avoid conflicts.
        print(f'Initiating vectordb {db_uuid}.')
        self.db_: Chroma = Chroma(
            embedding_function=embedding,
            collection_name=db_uuid,
        )

    def add_documents(self, documents: list[Document]):
        num_docs = len(documents)
        self.num_docs_ += num_docs
        print(f'Adding {num_docs} documents into database.')
        self.db_.add_documents(documents=documents)

    def retrieve(self,
                 query: str,
                 num_retrieval: int | None = None,
                 score_threshold: float = 0.5) -> List[Document]:
        """
        Search for documents related to a query using text embeddings and cosine distance.

        Args:
            query (str): The query string to search for related documents.
            num_retrieval (int): The max number of docs to retrieve based on similarity.
                If None, then use sqrt(num_docs) based on Plato distribution assumption.
            score_threshold (float): The score between 0 to 1 to filter the retrieved docs when greater.

        Returns:
            List[Document]: A list of Document objects representing the related documents found.
        """
        print(f'Searching for related works of: {query}...')
        # Default number of retrieval is set to be sqrt(num_docs) based on the assumption that the important docs is in Plato distribution.
        if not num_retrieval:
            num_retrieval = int(math.sqrt(self.num_docs_))
            print(f"Using the default num_retrieval = {num_retrieval} from totally {self.num_docs_} docs based on the Plato distribution assumption.")

        sources: List[Document] = self.db_.similarity_search_with_relevance_scores(
            query=query, 
            k=num_retrieval,
            score_threshold=score_threshold,
        )
        print(f'{len(sources)} sources found.')
        return sources
