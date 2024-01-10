import uuid
from app.web.db import db
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from .base import BaseModel


class Message(BaseModel):
    id: str = db.Column(
        db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_on = db.Column(db.DateTime, server_default=db.func.now())
    role: str = db.Column(db.String(), nullable=False)
    content: str = db.Column(db.String(), nullable=False)

    conversation_id: str = db.Column(
        db.String(), db.ForeignKey("conversation.id"), nullable=False
    )
    conversation = db.relationship("Conversation", back_populates="messages")

    def as_dict(self):
        return {"id": self.id, "role": self.role, "content": self.content}

    def as_lc_message(self) -> HumanMessage | AIMessage | SystemMessage:
        if self.role == "human":
            return HumanMessage(content=self.content)
        elif self.role == "ai":
            return AIMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        else:
            raise Exception(f"Unknown message role: {self.role}")
