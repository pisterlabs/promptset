import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from telebot.types import Message

load_dotenv()
embedding_function = embedding_functions.CohereEmbeddingFunction(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-english-v3.0",
)


class MessageDB:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient()
        self.collection = self._client.get_or_create_collection(
            name="message_history",
            embedding_function=embedding_function,
        )

    def add_message(self, message: Message) -> None:
        if message.content_type != "text":
            return

        self.collection.add(
            ids=str(message.id),
            metadatas={
                "chat_id": message.chat.id,
                "is_bot": message.from_user.is_bot,
            },
            documents=message.text,
        )

    def get_all_history(self, message: Message) -> List[BaseMessage]:
        docs = self.collection.get(
            where={"chat_id": message.chat.id}, include=["metadatas", "documents"]
        )
        return [
            AIMessage(content=docs["documents"][i])
            if docs["metadatas"][i]["is_bot"]
            else HumanMessage(content=docs["documents"][i])
            for i in sorted(range(len(docs["ids"])), key=lambda j: int(docs["ids"][j]))
        ]

    def get_relevant_history(
        self, message: Message, n_results: int = 3
    ) -> List[BaseMessage]:
        # TODO: Determine ordering and whether to use role tags
        if message.content_type != "text":
            return []

        docs = self.collection.query(
            query_texts=message.text,
            where={"chat_id": message.chat.id},
            n_results=n_results,
            include=["metadatas", "documents"],
        )
        return [
            AIMessage(content=content)
            if metadata["is_bot"]
            else HumanMessage(content=content)
            for metadata, content in zip(docs["metadatas"], docs["documents"])
        ]

    def clear_history(self, message: Message) -> None:
        self.collection.delete(where={"chat_id": message.chat.id})
