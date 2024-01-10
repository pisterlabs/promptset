from typing import List
from enum import Enum
from pydantic import BaseModel
from beanie import Document
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate, AIMessagePromptTemplate
from ..config import settings


class ChatRole(Enum):
    ai = 0,
    user = 1,


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSession(BaseModel):
    session_id: str
    session_title: str = 'New Chat with AI'
    messages: List[ChatMessage] = []

    def add_ai_message(self, content: str):
        self.messages.append(ChatMessage(
            role=ChatRole.ai.name, content=content))

    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(
            role=ChatRole.user.name, content=content))

    def get_chat_prompt_template(self, system_instruction: str):
        return ChatPromptTemplate.from_messages([SystemMessage(
            content=system_instruction)] + [AIMessagePromptTemplate.from_template(msg.content) if msg.role == ChatRole.ai.name
                                            else HumanMessagePromptTemplate.from_template(msg.content) for msg in self.messages])

    def get_latest_response(self):
        return self.messages[-1].content if self.messages else ""

    class Settings:
        name = settings.DATABASE_NAME

    class Config:
        populate_by_name = True


class UserSecurityQuestion(BaseModel):
    question: str
    response: str


class User(Document):
    full_name: str
    email: str
    hashed_password: str
    session_list: List[ChatSession] = []
    security_question: UserSecurityQuestion
