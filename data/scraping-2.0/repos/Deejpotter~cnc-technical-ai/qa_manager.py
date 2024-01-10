import uuid
import openai
from IDataManager import IDataManager
from mongo_data_manager import MongoDataManager
from pinecone_data_manager import PineconeDataManager


class QAManager(IDataManager):
    """
    QAManager class handles the interaction with a Pinecone index for storing and retrieving QA pairs.
    It manages the creation of vector embeddings and vector search functionality.
    Implements the IDataManager interface.
    """

    def __init__(self):
        """
        Initialize the DataManager with the Pinecone configuration and get a reference to the QA index.
        """
        self.pinecone_data_manager = PineconeDataManager("cnctechnicalai")

    def create(self, question, answer):
        """
        Add a QA pair with both text and vector representations.

        Args:
            question (str): The question text.
            answer (str): The answer text.
        """
        qa_id = str(uuid.uuid4())  # Generate a unique identifier
        embeddings = self.create_vector_embeddings(question)
        data = {
            "id": qa_id,
            "vector": embeddings,
            "metadata": {"question": question, "answer": answer},
        }
        self.pinecone_data_manager.create(data)

    def get(self, question):
        """
        Fetch a QA pair by its question.

        Args:
            question (str): The question text as the ID.
        """
        return self.pinecone_data_manager.get(question)

    def update(self, question, answer):
        """
        Update a QA pair in the Pinecone index.

        Args:
            question (str): The question text as the ID.
            answer (str): The new answer text.
        """
        question_vector = self.create_vector_embeddings(question)
        self.pinecone_data_manager.update(
            question, {"vector": question_vector, "metadata": {"answer": answer}}
        )

    def delete(self, question):
        """
        Delete a QA pair from the Pinecone index.

        Args:
            question (str): The question text as the ID.
        """
        self.pinecone_data_manager.delete(question)

    def find(self, query_text, top_k=10):
        """
        Perform a vector search to find similar questions.

        Args:
            query_text (str): The query text for searching similar questions.
            top_k (int): Number of top results to return.
        """
        query_vector = self.create_vector_embeddings(query_text)
        return self.pinecone_data_manager.find(query_vector, top_k)

    def create_vector_embeddings(self, text: str) -> list:
        """
        Generate embeddings for the given text using OpenAI API.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            list: The generated embeddings as a list of floats.
        """
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
