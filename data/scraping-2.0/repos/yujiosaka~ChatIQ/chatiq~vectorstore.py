import logging
from typing import Any, List

from langchain.docstore.document import Document
from langchain.vectorstores import Weaviate
from weaviate import Client
from weaviate.exceptions import WeaviateBaseError

from chatiq.document_loaders import DummyMessageLoader
from chatiq.retriever import Retriever


class Vectorstore:
    """Manages interaction with Weaviate indexes for storing, retrieving, and removing Slack messages.

    The Vectorstore uses Weaviate. It maintains a separate Weaviate class (analogous to an index) for each team.

    Args:
        weaviate_client (Client): A client instance for interacting with the Weaviate server.
        team_id (str): The Slack team ID. This is used to create a separate index for each team.

    Attributes:
        logger (Logger): A logger instance to log debug and error messages.
        weaviate_client (Client): A client instance for interacting with the Weaviate server.
        team_id (str): The Slack team ID.
        index_name (str): The name of the Weaviate class (index) associated with the team.
        weaviate (Weaviate): A Weaviate instance to interact with the Weaviate class.
    """

    INDEX_KEY = "content"
    ATTRIBUTES = [
        "file_or_attachment_id",
        "content_type",
        "channel_type",
        "channel_id",
        "thread_ts",
        "ts",
        "permalink",
        "timestamp",
    ]
    CLASS_SCHEMA = {
        "class": "<class to be updated>",
        "description": "<description to be updated>",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {"text2vec-transformers": {"poolingStrategy": "masked_mean", "vectorizeClassName": False}},
        "properties": [
            {
                "dataType": ["text"],
                "description": "The content of the message",
                "moduleConfig": {"text2vec-transformers": {"skip": False, "vectorizePropertyName": False}},
                "name": "content",
            },
            {
                "dataType": ["string"],
                "description": "The file or attachment ID",
                "name": "file_or_attachment_id",
            },
            {
                "dataType": ["string"],
                "description": "The content type",
                "name": "content_type",
            },
            {
                "dataType": ["string"],
                "description": "The channel type",
                "name": "channel_type",
            },
            {
                "dataType": ["string"],
                "description": "The channel ID",
                "name": "channel_id",
            },
            {
                "dataType": ["string"],
                "description": "The message ID",
                "name": "ts",
            },
            {
                "dataType": ["string"],
                "description": "The thread ID",
                "name": "thread_ts",
            },
            {
                "dataType": ["string"],
                "description": "The permalink",
                "name": "permalink",
            },
            {
                "dataType": ["date"],
                "description": "The timestamp of the message",
                "name": "timestamp",
            },
        ],
    }

    def __init__(self, weaviate_client: Client, team_id: str):
        """Initializes a Vectorstore instance for a specific team.

        Args:
            weaviate_client (Client): A Weaviate client instance used to interact with Weaviate.
            team_id (str): The ID of the team for which the Vectorstore instance is created.
        """

        self.logger = logging.getLogger(__name__)
        self.weaviate_client = weaviate_client
        self.team_id = team_id
        self.index_name = f"Message{self.team_id}"
        self.weaviate = Weaviate(self.weaviate_client, self.index_name, self.INDEX_KEY, None, self.ATTRIBUTES)

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Adds a list of documents to the Weaviate index associated with the team.

        Args:
            documents (List[Document]): A list of Document objects to be added to the index.

        Returns:
            List[str]: A list of document IDs for the documents that were added.
        """

        self.logger.debug(f"Adding {len(documents)} documents to {self.index_name} index for Weaviate")

        return self.weaviate.add_documents(documents, **kwargs)

    def delete_message(self, ts: str) -> None:
        """Deletes a message from the Weaviate class (index) associated with the team.

        Args:
            ts (str): The ID of the message to be deleted from the index.

        Raises:
            WeaviateBaseError: If there is any error while deleting the document.
        """

        self.logger.debug(f"Deleteing message from {self.index_name} index for Weaviate")

        try:
            with self.weaviate_client.batch as batch:
                batch.delete_objects(
                    class_name=self.index_name,
                    where={"path": ["ts"], "operator": "Equal", "valueString": ts},
                )
        except Exception as e:
            error_message = f"Failed to delete message from {self.index_name} index for Weaviate. Error: {e}"
            self.logger.error(error_message)
            raise WeaviateBaseError(error_message)

        self.logger.info(f"Deleted message from {self.index_name} index for Weaviate")

    def delete_file_or_attachment(self, file_or_attachment_id: str) -> None:
        """Deletes a file or attachment from the Weaviate class (index) associated with the team.

        Args:
            file_or_attachment_id (str): The ID of the file or attachment to be deleted from the index.

        Raises:
            WeaviateBaseError: If there is any error while deleting the document.
        """

        self.logger.debug(f"Deleteing file or attachment from {self.index_name} index for Weaviate")

        try:
            with self.weaviate_client.batch as batch:
                batch.delete_objects(
                    class_name=self.index_name,
                    where={"path": ["file_or_attachment_id"], "operator": "Equal", "valueString": file_or_attachment_id},
                )
        except Exception as e:
            error_message = f"Failed to delete file or attachment from {self.index_name} index for Weaviate. Error: {e}"
            self.logger.error(error_message)
            raise WeaviateBaseError(error_message)

        self.logger.info(f"Deleted file or attachment from {self.index_name} index for Weaviate")

    def delete_channel(self, channel_id: str) -> None:
        """Deletes a channel from the Weaviate class (index) associated with the team.

        Args:
            channel_id (str): The ID of the channel to be deleted from the index.

        Raises:
            WeaviateBaseError: If there is any error while deleting the document.
        """

        self.logger.debug(f"Deleteing channel from {self.index_name} index for Weaviate")

        try:
            with self.weaviate_client.batch as batch:
                batch.delete_objects(
                    class_name=self.index_name,
                    where={"path": ["channel_id"], "operator": "Equal", "valueString": channel_id},
                )
        except Exception as e:
            error_message = f"Failed to delete channel from {self.index_name} index for Weaviate. Error: {e}"
            self.logger.error(error_message)
            raise WeaviateBaseError(error_message)

        self.logger.info(f"Deleted channel from {self.index_name} index for Weaviate")

    def as_retriever(self, is_private: bool, channel_id: str, thread_ts: str) -> Retriever:
        """Returns a retriever for fetching documents from the Weaviate index based on certain conditions.

        The conditions for retrieval vary based on whether the channel is private or public.

        Args:
            is_private (bool): True if the channel is private, False otherwise.
            channel_id (str): The ID of the channel where the bot was mentioned.
            thread_ts (str): The ID of the thread. This is used to exclude messages in the current thread.

        Returns:
            Retriever: A retriever object for fetching documents from the index.
        """

        return Retriever(self.weaviate, is_private, channel_id, thread_ts)

    def ensure_index(self):
        """Ensures that the Weaviate class (index) for the team exists.

        If the index does not exist, it is created.

        Raises:
            WeaviateBaseError: If there is any error while checking or creating the index.
        """

        self.logger.debug(f"Ensuring {self.index_name} index for Weaviate")

        try:
            if not self.weaviate_client.schema.exists(self.index_name):
                class_schema = {
                    **self.CLASS_SCHEMA,
                    "class": self.index_name,
                    "description": f"A Slack message index for {self.team_id}",
                }
                self.weaviate_client.schema.create({"classes": [class_schema]})

                # LangChain throws index out of range error when vectorstore has no document to query against.
                # Add dummy documents to workaround the issue
                self.weaviate.add_documents(DummyMessageLoader().load())
                self.logger.info(f"Created {self.index_name} index for Weaviate")
        except Exception as e:
            error_message = f"Failed to ensure {self.index_name} index from Weaviate. Error: {e}"
            self.logger.error(error_message)
            raise WeaviateBaseError(error_message)

    def delete_index(self):
        """Deletes the Weaviate class (index) associated with the team.

        Raises:
            WeaviateBaseError: If there is any error while deleting the index.
        """

        self.logger.debug(f"Removing {self.index_name} index from Weaviate")

        try:
            if self.weaviate_client.schema.exists(self.index_name):
                self.weaviate_client.schema.delete_class(self.index_name)
                self.logger.info(f"Removed {self.index_name} index from Weaviate")
        except Exception as e:
            error_message = f"Failed to remove {self.index_name} index from Weaviate. Error: {e}"
            self.logger.error(error_message)
            raise WeaviateBaseError(error_message)
