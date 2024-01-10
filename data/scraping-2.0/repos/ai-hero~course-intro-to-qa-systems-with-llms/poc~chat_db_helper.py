"""
    Helper functions for chat database.
    We keep the DB related and embedding related functions common, so that we can use the same
    functions for data preparation and data querying chatbot.
"""

import os
from typing import Any, Dict, List, Optional

import chromadb
import openai
from tqdm import tqdm

openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_embedding(text: str, embeddings_cache: Optional[Dict[str, List[float]]] = None) -> Any:
    """Use the same embedding generator as what was used on the data!!!"""
    if embeddings_cache and text in embeddings_cache:
        return embeddings_cache[text]
    if len(text) / 4 > 8000:  # Hack to be under the 8k limit, one token ~= 4 characters
        text = text[:8000]
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding


class Chat:
    """A chat conversation."""

    def __init__(
        self,
        thread_id: str,
        text: str,
        summary: Optional[str] = None,
        embeddings_cache: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Initialize the chat."""
        self.thread_id = thread_id
        self.text = text
        if summary:
            self.summary = summary
        else:
            self.summary = self._summarize(text)
        self.embedding = get_embedding(text, embeddings_cache)

    def _summarize(self, text: str) -> Any:
        """Summarize conversations since individually they are long and go over 8k limit"""
        if len(text) / 4 > 3800:  # Hack to be under the 4k limit, one token ~= 4 characters
            text = text[:3800]
        prompt = (
            "Summarize the following Slack conversation. Do not use ids, usernames, mentions, \
and links in the summary. \
If there is a question asked, please include the question, and the summarized answer. \
If there is no question, generate a question that might be used to retrieve this summary.```"
            + text
            + "```"
        )
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content

    def __repr__(self) -> str:
        return f"ChatDocument(thread_id={self.thread_id})"


class ChatVectorDB:
    """A vector database for chat conversations."""

    chat_db_version = "01"  # Version of the chat database

    def __init__(self) -> None:
        """Initialize the database."""
        self.db_name = "chroma.db"
        self.collection_name = f"chats-{ChatVectorDB.chat_db_version}"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, ".content")
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(os.path.join(db_path, "chroma.db"))
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except ValueError:
            # Create the collection if it doesn't exist
            self.collection = self.client.create_collection(self.collection_name)

    def search_index(self, text: str) -> List[Chat]:
        """Search the index for the top 3 results."""
        embedding = get_embedding(text)
        results = self.collection.query(query_embeddings=[embedding], n_results=3)
        return [
            Chat(thread_id=m["thread_id"], text=m["text"], summary=m.get("summary", ""))
            for m in results["metadatas"][0]
        ]

    def add(self, chats: List[Chat]) -> None:
        """Add the embeddings, metadata, and ids to the database."""

        embeddings_list = []
        metadata_list = []
        id_list = []

        for chat in tqdm(chats):
            metadata = {
                "thread_id": chat.thread_id,
                "text": chat.text,
                "summary": chat.summary,
            }
            metadata_list.append(metadata)
            embeddings_list.append(chat.embedding)
            id_list.append(chat.thread_id)

        self.collection.add(embeddings=embeddings_list, metadatas=metadata_list, ids=id_list)
