# import json
import logging
from typing import List

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)
from pymongo.errors import OperationFailure
from pymongo.mongo_client import MongoClient

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_STRING = (
    "mongodb://root:root@mongo:27017/messages?authSource=admin"
)


class MongoChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        connection_string: str = DEFAULT_CONNECTION_STRING,
        collection_name: str = "message_store",
    ):
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client.get_default_database()
        except OperationFailure as error:
            logger.error(error)

        self.session_id = session_id
        self.collection = self.db.get_collection(collection_name)

        # self._create_collection_if_not_exists()

    def ping(self):
        try:
            self.client.admin.command("ping")
            print("You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    # TODO: ↓多分要らない
    def _create_collection_if_not_exists(self) -> None:
        create_collection_query = {
            "create": self.table_name,
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["session_id", "message"],
                    "properties": {
                        "id": {"bsonType": "objectId"},
                        "session_id": {"bsonType": "string"},
                        "message": {"bsonType": "object"},
                    },
                }
            },
        }
        self.db.command(create_collection_query)

    @property
    def messages(self) -> List[BaseMessage]:
        """
        Retrieve the messages from MongoDB
        """
        items = [
            record["message"]
            for record in self.collection.find({"session_id": self.session_id})
        ]
        messages = messages_from_dict(items)
        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: str) -> None:
        """
        Append the message to the record in MongoDB
        """
        self.collection.insert_one(
            {
                "session_id": self.session_id,
                "message": _message_to_dict(message),
            }
        )

    def clear(self) -> None:
        """
        Clear session memory from MongoDB
        """
        self.collection.delete_many({"session_id": self.session_id})

    def __del__(self) -> None:
        if self.client:
            self.client.close()
