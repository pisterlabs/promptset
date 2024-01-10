from loguru import logger
from collections import defaultdict
from spiritualdata_utils import mongo_connect_db
import uuid
from cachetools.func import ttl_cache
from bson.objectid import ObjectId
import json
from datetime import datetime, MINYEAR
from typing import List
from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema.messages import (
    BaseMessage,
    _message_to_dict,
    messages_from_dict,
    _message_from_dict,
)
from pymongo import MongoClient, errors
from bson.errors import InvalidId
from fastapi import HTTPException
import os


DEFAULT_DBNAME = "user_interaction"
DEFAULT_COLLECTION_NAME = "chats"


# Cache in memory all used chat histories for 1 hour
@ttl_cache(maxsize=1, ttl=3600)
def get_chat_history_manager():
    mongo = mongo_connect_db(
        uri=os.getenv("MONGOURI_CHATS", os.getenv("MONGOURI")),
        database_name=DEFAULT_DBNAME,
    )
    chat_history_manager = ChatHistoryManager(
        mongo_collection=mongo[DEFAULT_COLLECTION_NAME]
    )
    return chat_history_manager


class ChatHistoryManager:
    def __init__(self, mongo_collection):
        logger.info(
            f"About to init chat history manager for collection {mongo_collection}"
        )
        self.mongo_collection = mongo_collection
        self.loaded_chats = {}
        self.mongo_collection.create_index("user_id")
        logger.info(
            f"Initialized chat history manager for collection {self.mongo_collection}"
        )

    def get_chat_history(self, user_id: str, chat_id: str = None):
        logger.info(
            f"About to get chat history for user_id: {user_id} chat_id: {chat_id}"
        )
        if isinstance(chat_id, str):
            chat_id = chat_id.strip()

        if not chat_id:
            logger.info("about to insert new chat")
            inserted = self.mongo_collection.insert_one(
                {"user_id": user_id, "messages": []}
            )
            chat_id = str(inserted.inserted_id)
            logger.info(f"Got new chat_id: {inserted.inserted_id}")
        if (user_id, chat_id) not in self.loaded_chats:
            self.loaded_chats[(user_id, chat_id)] = self._load_chat_history(
                user_id, chat_id
            )
        logger.info(f"got chat_history for user_id: {user_id} chat_id: {chat_id}")
        return self.loaded_chats[(user_id, chat_id)], chat_id

    def _load_chat_history(self, user_id: str, chat_id: str):
        logger.info(
            f"About to load chat history for user_id: {user_id} chat_id: {chat_id}"
        )
        try:
            document = self.mongo_collection.find_one(
                {"_id": ObjectId(chat_id), "user_id": user_id}, {"history": 1}
            )
        except InvalidId as e:
            logger.error(e)
            raise HTTPException(
                status_code=400, detail="Invalid chat_id format: " + str(chat_id)
            )
        if document:
            # If a chat history exists in MongoDB, load it into a HybridChatHistory object
            history = document.get("history", [])
            chat_history = HybridChatHistory(user_id, chat_id, self.mongo_collection)
            for message in history:
                message_object = _message_from_dict(message)
                chat_history.chat_message_history.add_message(message_object)
        else:
            # If no chat history exists for this user and chat_id, create a new one
            chat_history = HybridChatHistory(user_id, chat_id, self.mongo_collection)
        return chat_history

    def get_user_chats(self, user_id: str):
        # Fetch all documents belonging to the user
        documents = self.mongo_collection.find({"user_id": user_id})

        # Function to extract 'updated_at' or set a default very old date if it doesn't exist
        def get_updated_at(doc):
            return doc.get("updated_at", datetime(MINYEAR, 1, 1))

        # Sort documents by 'updated_at' descending (most recent first)
        # and handle cases where 'updated_at' might be missing
        sorted_documents = sorted(documents, key=get_updated_at, reverse=True)

        # Convert each sorted document into a dict with id and title
        user_chats = {
            str(doc["_id"]): {"title": doc.get("title")} for doc in sorted_documents
        }
        logger.info(f"Got user chats: {len(user_chats)} for user_id: {user_id}")
        return user_chats

    def delete_chat(self, user_id: str, chat_id: str):
        # Delete chat from MongoDB
        result = self.mongo_collection.delete_one(
            {"_id": ObjectId(chat_id), "user_id": user_id}
        )

        # Delete chat from loaded_chats
        if (user_id, chat_id) in self.loaded_chats:
            del self.loaded_chats[(user_id, chat_id)]

        return (
            result.deleted_count > 0
        )  # Return True if a document was deleted, False otherwise


"""
Custom MongoDB message history class
Reference: https://github.com/hwchase17/langchain/blob/master/langchain/memory/chat_message_histories/mongodb.py
"""


def chat_history_to_str(chat_history, max_per_message=1000, max_messages=2):
    chat_history_str = ""
    if chat_history and hasattr(chat_history, "messages"):
        for i, message in enumerate(chat_history.messages):
            content = message.content[:max_per_message]
            chat_history_str += f"{message.type}: {content}" + "\n"
            if i + 1 >= max_messages:
                break
    return chat_history_str


class HybridChatHistory(BaseChatMessageHistory):
    """
    Used to store chat history and title in MongoDB by the end of the API call,
    and also retains chats in-memory for as long as ChatHistoryManager remains in cache, which means relatively few reads from Mongo (only for old chats).
    """

    def __init__(self, user_id, chat_id, mongo_collection):
        self.user_id = user_id
        self.chat_id = chat_id
        self.collection = mongo_collection
        self.chat_message_history = ChatMessageHistory()
        self.messages_buffer = []

    @property
    def messages(self):
        return self.chat_message_history.messages

    def add_message(self, message):
        """
        Add message to the in-memory chat history and to the buffer
        """
        self.chat_message_history.add_message(message)
        self.messages_buffer.append(_message_to_dict(message))

    def save_chat(self, title=None):
        """
        Only now will the chat history actually be saved to MongoDB.
        """
        # Push all buffered messages to MongoDB and update the title field
        prev_title = self.collection.find_one(
            {"user_id": self.user_id, "_id": ObjectId(self.chat_id)}
        ).get("title")

        if not prev_title:
            self.collection.update_one(
                {"user_id": self.user_id, "_id": ObjectId(self.chat_id)},
                {"$set": {"title": title}},
                upsert=True,
            )

        self.collection.update_one(
            {"user_id": self.user_id, "_id": ObjectId(self.chat_id)},
            {
                "$push": {"history": {"$each": self.messages_buffer}},
                "$set": {"updated_at": datetime.utcnow()},
            },
            upsert=True,
        )
        # Clear the buffer
        self.messages_buffer.clear()

    def clear(self):
        self.chat_message_history.clear()
        try:
            self.collection.delete_many(
                {"user_id": self.user_id, "chat_id": ObjectId(self.chat_id)}
            )
        except errors.WriteError as err:
            logger.error(err)
