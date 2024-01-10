import json
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pybot.utils import utcnow


class File(BaseModel):
    id: Optional[str] = None
    filename: str
    path: str
    size: int = 0

    @model_validator(mode="before")
    @classmethod
    def set_id_and_path(cls, values):
        if "pk" in values and "id" not in values:
            values["id"] = values["pk"]
        # I don't want to expose the full path to the user or LLM, so I use a mounted path instead.
        # It may looks weird, but using alias sucks.
        if "mounted_path" in values:
            values["path"] = values["mounted_path"]
        return values


class ChatMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(default_factory=uuid4)
    """Message id, used to chain stream responses into message."""
    conversation: Optional[str] = None
    """Conversation id"""
    from_: Optional[str] = Field(None, alias="from")
    """A transient field to determine conversation id."""
    content: Optional[str | File] = None
    type: Literal[
        "text", "stream/start", "stream/text", "stream/end", "file", "info", "error"
    ] = "text"
    feedback: Literal["thumbup", "thumbdown", None] = None
    additional_kwargs: Optional[dict[str, Any]] = None
    # sent_at is not an important information for the user, as far as I can tell.
    # But it introduces some complexity in the code, so I'm removing it for now.
    # sent_at: datetime = Field(default_factory=datetime.now)

    @staticmethod
    def from_lc(
        lc_message: BaseMessage, conv_id: str, from_: str = None
    ) -> "ChatMessage":
        """Convert from langchain message.
        Note: for file messages, the content is used for LLM, and other fields are used for displaying to frontend.

        Args:
            lc_message (BaseMessage): _description_
            conv_id (str): _description_
            from_ (str, optional): _description_. Defaults to None.

        Returns:
            ChatMessage: _description_
        """
        msg_id_str = lc_message.additional_kwargs.get("id", None)
        msg_id = UUID(msg_id_str) if msg_id_str else uuid4()
        msg_type = lc_message.additional_kwargs.get("type", "text")
        if msg_type == "file":
            msg_content = File.model_validate(
                {
                    "id": lc_message.additional_kwargs.get("file_id", None),
                    "size": lc_message.additional_kwargs.get("size", None),
                    **json.loads(lc_message.content),
                }
            )
        else:
            msg_content = lc_message.content
        return ChatMessage(
            id=msg_id,
            conversation=conv_id,
            from_=from_ if from_ else lc_message.type,
            content=msg_content,
            type=msg_type,
            feedback=lc_message.additional_kwargs.get("feedback", None),
        )

    def to_lc(self) -> BaseMessage:
        """Convert to langchain message.
        Note: for file messages, the content is used for LLM, and other fields are used for displaying to frontend.
        """
        additional_kwargs = {
            "id": str(self.id),
            "type": self.type,
        }
        if self.type == "file":
            content = json.dumps(
                {
                    "filename": self.content.filename,
                    "path": self.content.path,
                }
            )
            additional_kwargs = additional_kwargs | {
                "file_id": str(self.content.id),
                "size": self.content.size,
            }
        else:
            content = self.content
        match self.from_:
            case "system":
                return SystemMessage(
                    content=content,
                    additional_kwargs=additional_kwargs,
                )
            case "ai":
                return AIMessage(
                    content=content,
                    additional_kwargs=additional_kwargs,
                )
            case _:  # username
                return HumanMessage(
                    content=content,
                    additional_kwargs=additional_kwargs,
                )

    def model_dump(
        self, by_alias: bool = True, exclude_none: bool = True, **kwargs
    ) -> dict[str, Any]:
        return super().model_dump(
            by_alias=by_alias, exclude_none=exclude_none, **kwargs
        )

    def model_dump_json(
        self, by_alias: bool = True, exclude_none: bool = True, **kwargs
    ) -> str:
        return super().model_dump_json(
            by_alias=by_alias, exclude_none=exclude_none, **kwargs
        )


class InfoMessage(ChatMessage):
    content: dict[str, Any]
    type: Literal["info"] = "info"


class Conversation(BaseModel):
    id: Optional[str] = None
    title: str
    owner: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = created_at

    @model_validator(mode="before")
    @classmethod
    def set_id(cls, values):
        if "pk" in values and "id" not in values:
            values["id"] = values["pk"]
        return values


class ConversationDetail(Conversation):
    """Conversation with messages."""

    messages: list[ChatMessage] = []


class CreateConversation(BaseModel):
    title: str


class UpdateConversation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str


class UserProfile(BaseModel):
    userid: str
    username: Optional[str] = None
    email: Optional[str] = None
