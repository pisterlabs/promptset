from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from langchain.schema import AgentAction, BaseMessage
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from sqlbot.utils import utcnow


class IntermediateStep(RootModel):
    """Used to serialize intermediate step.
    Python's tuple serializes as a list ('[elem1, elem2]'), making deserialization challenging.
    To address this, I opt to store it as a list.

    Additionally, langchain.schema.AgentAction is a pydantic v1 model, which proves somewhat verbose
    to serialize and deserialize in v2. Consequently I store it as a dict.

    While there is consideration storing action and observation as different fields in the future,
    for alignment with langchain's schema (also for simplicity), I currently store them as a list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: list[Any]
    """[dict, Any] of AgentAction and observation.
    I don't know why deserializing tuple is such a pain in the ass, list just works.
    """

    def __len__(self):
        return len(self.root)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    @model_validator(mode="before")
    @classmethod
    def tuple_to_list(cls, values):
        if isinstance(values, tuple) or isinstance(values, list):
            action = values[0]
            if isinstance(action, AgentAction):
                return [action.dict(), values[1]]
            return [action, values[1]]
        return values


class IntermediateSteps(RootModel):
    """Used to serialize list of pydantic models."""

    root: list[IntermediateStep]

    def __len__(self):
        return len(self.root)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


class ChatMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    """Message id, used to chain stream responses into message."""
    conversation: Optional[str] = None
    """Conversation id"""
    from_: Optional[str] = Field(None, alias="from")
    """A transient field to determine conversation id."""
    content: Optional[str] = None
    type: str
    intermediate_steps: Optional[IntermediateSteps] = None
    # sent_at is not an important information for the user, as far as I can tell.
    # But it introduces some complexity in the code, so I'm removing it for now.
    # sent_at: datetime = Field(default_factory=datetime.now)

    @staticmethod
    def from_lc(lc_message: BaseMessage, conv_id: str, from_: str) -> "ChatMessage":
        msg_id_str = lc_message.additional_kwargs.get("id", None)
        msg_id = UUID(msg_id_str) if msg_id_str else uuid4()
        steps_str = lc_message.additional_kwargs.get("intermediate_steps", None)
        steps = IntermediateSteps.model_validate_json(steps_str) if steps_str else None
        return ChatMessage(
            id=msg_id,
            conversation=conv_id,
            from_=from_,
            content=lc_message.content,
            type="text",
            intermediate_steps=steps,
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


class UpdateConversation(BaseModel):
    title: str
