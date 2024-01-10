"""
A custom loader for Piazza data.

TODO(michaelfromyeg): eventually introduce new loaders, to interact with all course data. (e.g., textbooks)

Here, I will treat Piazza as a chat platform, and tell the LLM to act as the answer-er.
"""
import json
import os
from typing import Iterator

from langchain_community.chat_loaders.base import BaseChatLoader
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage

CWD = os.getcwd()


class PiazzaLoader(BaseChatLoader):
    """
    A custom loader for Piazza data.
    """

    def __init__(self, course: str):
        self.course = course

    def _load_conversation(self, path: str) -> ChatSession:
        """
        Load a single conversation from a path.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(data)

        metadata = data.get("metadata", {})
        is_follow_up = metadata.get("is_follow_up", False)

        question = HumanMessage(
            content=data.get("question", "") or "",
            additional_kwargs={"subject": data.get("subject", "") or ""},
        )

        answer = HumanMessage(
            content=data.get("answer", "") or "",
            additional_kwargs={
                "is_follow_up": is_follow_up or False,
                "upvotes": metadata.get("upvotes", 0) or "",
                "original_question_id": metadata.get("original_question_id", "") or "",
            },
        )

        chat_session = ChatSession(messages=[question, answer])

        return chat_session

    def lazy_load(self) -> Iterator[ChatSession]:
        """
        Load Piazza data.
        """
        course_path = os.path.join(CWD, "transformed_data", self.course)

        for instance in os.listdir(course_path):
            for conversation in os.listdir(os.path.join(course_path, instance)):
                try:
                    yield self._load_conversation(
                        os.path.join(course_path, instance, conversation)
                    )
                except Exception as e:
                    print(f"Failed to load conversation due to {e}")
                    continue
