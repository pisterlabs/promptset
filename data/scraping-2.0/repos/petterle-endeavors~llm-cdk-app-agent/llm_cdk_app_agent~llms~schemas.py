"""Define schemas for the generative endpoints."""
# pylint: disable=no-self-argument, duplicate-code
# we are disabling no-self because we are still using pydantic v1
# we are disabling duplicate-code because the api layer has very similar code
# to the backend layer
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Type, Union
from uuid import UUID, uuid4

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.messages import BaseMessageChunk
from pydantic.v1 import BaseModel, Field, validator


# we need to continue using pydantic v1 for now
# as we are inheriting from langchain.schema
# which is still using pydantic v1
# see https://python.langchain.com/docs/guides/pydantic_compatibility#:~:text=LangChain%20Pydantic%20migration%20plan%E2%80%8B,will%20continue%20to%20use%20V1.
class PydanticV1BaseModel(BaseModel):
    """Define the base model for schemas."""

    class Config:
        """Define the configuration for the model."""

        arbitrary_types_allowed = True


class ModelName(str, Enum):
    """Define the supported LLMs."""

    GPT_TURBO = "gpt-3.5-turbo"
    GPT_TURBO_LARGE_CONTEXT = "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4"


MODEL_NAME_TO_MODEL_CLASS: Dict[ModelName, Type[BaseChatModel]] = {
    ModelName.GPT_TURBO: ChatOpenAI,
    ModelName.GPT_TURBO_LARGE_CONTEXT: ChatOpenAI,
    ModelName.GPT_4: ChatOpenAI,
}


def get_model(
    llm_model_name: ModelName,
    api_key: str,
    stream: bool = False,
) -> BaseChatModel:
    """
    Get an instance of a chat model.

    Args
    ----
        llm_model_name: The name of the LLM model to use.
        api_key: The API key for the LLM model.
        stream: Whether or not to stream the response.

    Returns
    -------
        An instance of the chat model.

    """
    # disabling the invalid name linting error because the model name is not snake case
    Model = MODEL_NAME_TO_MODEL_CLASS.get(llm_model_name)  # pylint: disable=invalid-name
    if not Model:
        raise ValueError(f"Unsupported model name: {llm_model_name}")
    if issubclass(Model, ChatOpenAI):
        model = Model(
            model=llm_model_name,
            streaming=stream,
            openai_api_key=api_key,
        )
    else:
        raise NotImplementedError(f"Unsupported model: {Model}")
    return model


class ChatRole(str, Enum):
    """Define the built-in MongoDB roles."""

    AI = "assistant"
    USER = "user"
    FUNCTION = "function"
    SYSTEM = "system"


class BaseChat(BaseMessage):
    """Define the model for the chat message."""

    role: ChatRole = Field(
        ...,
        description="The role of the creator of the chat message.",
    )
    id: UUID = Field(
        default_factory=UUID,
        description="The ID of the chat message.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="The timestamp of the chat message.",
    )
    render_chat: bool = Field(
        default=True,
        description="Whether or not to render the chat message. If false, the chat message MUST be hidden from the user.",
    )
    type: str = Field(
        default="base_chat",
        description="The type of the chat message.",
    )


class SystemChat(SystemMessage, BaseChat):
    """Define the model for the system chat message."""

    @validator("role")
    def role_must_be_system(cls, role: ChatRole) -> ChatRole:
        """Validate that the role is system."""
        if role != ChatRole.SYSTEM:
            raise ValueError("The role must be system.")
        return role

    @validator("render_chat")
    def render_chat_must_be_false(cls, render_chat: bool) -> bool:
        """Validate that render_chat is false."""
        if render_chat:
            raise ValueError("The render_chat must be false.")
        return render_chat


class FunctionChat(FunctionMessage, BaseChat):
    """Define the model for the function chat message."""

    @validator("role")
    def role_must_be_function(cls, role: ChatRole) -> ChatRole:
        """Validate that the role is function."""
        if role != ChatRole.FUNCTION:
            raise ValueError("The role must be function.")
        return role

    @validator("render_chat")
    def render_chat_must_be_false(cls, render_chat: bool) -> bool:
        """Validate that render_chat is false."""
        if render_chat:
            raise ValueError("The render_chat must be false.")
        return render_chat


class UserChat(HumanMessage, BaseChat):
    """Define the base model for the student chat message."""

    @validator("role")
    def role_must_be_user(cls, role: ChatRole) -> ChatRole:
        """Validate that the role is user."""
        if role != ChatRole.USER:
            raise ValueError("The role must be user.")
        return role


class FunctionCall(PydanticV1BaseModel):
    """Define the model for the AI response calling function."""

    name: str = Field(
        ...,
        description="The name of the function to call.",
    )
    arguments: dict = Field(
        ...,
        description="The arguments to pass to the function.",
    )


class AIChat(AIMessage, BaseChat):
    """Define the model for the TAI Tutor chat message."""

    function_call: Optional[FunctionCall] = Field(
        default=None,
        description="The function call that the assistant wants to make.",
    )

    @validator("role")
    def role_must_be_ai(cls, role: ChatRole) -> ChatRole:
        """Validate that the role is ai."""
        if role != ChatRole.AI:
            raise ValueError("The role must be ai.")
        return role


class ChatSession(PydanticV1BaseModel):
    """Define the request model for the chat endpoint."""

    create_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="The timestamp of the chat session creation.",
    )
    chats: List[Union[SystemChat, FunctionChat, AIChat, UserChat]] = Field(
        ...,
        description="The chat session message history.",
    )
    stream_response: bool = Field(
        default=False,
        description="Whether or not to stream the response.",
    )

    @property
    def last_chat(self) -> Optional[Union[FunctionChat, AIChat, UserChat]]:
        """Get the last chat message that isn't a system chat."""
        if self.chats and not isinstance(self.chats[-1], SystemChat):
            return self.chats[-1]
        return None

    def last_n_number_of_user_chats(self, n: int) -> List[UserChat]:  # pylint: disable=invalid-name
        """Get the last n number of user chats."""
        user_chats = []
        for chat in reversed(self.chats):
            if isinstance(chat, UserChat):
                user_chats.append(chat)
            if len(user_chats) == n:
                break
        return user_chats

    def append_chat(self, chat: Union[UserChat, AIChat, FunctionChat]):
        """Append a message to the chat session."""
        self.chats.append(chat)

    def append_message_chunk(self, message_chunk: BaseMessageChunk):
        """Append a message chunk to the chat session."""
        self.chats[-1].content += message_chunk.content

    def append_ai_function_call(self, function_call: FunctionCall) -> None:
        """Append a function call to the chat session."""
        ai_chat = AIChat(
            content="",
            role=ChatRole.AI,
            render_chat=False,
            id=uuid4(),
            function_call=function_call,
        )
        self.append_chat(ai_chat)

    def append_ai_chat(
        self,
        ai_message: str,
        render_chat: bool = True,
    ) -> None:
        """Append an AI response to the chat session."""
        ai_chat = AIChat(
            content=ai_message,
            role=ChatRole.AI,
            render_chat=render_chat,
            id=uuid4(),
        )
        self.append_chat(ai_chat)

    def append_function_response(self, function_name: str, response: str) -> None:
        """Append a function response to the chat session."""
        function_chat = FunctionChat(
            name=function_name,
            content=response,
            role=ChatRole.FUNCTION,
            id=uuid4(),
            render_chat=False,
        )
        self.append_chat(function_chat)

    def upsert_system_prompt(self, prompt: str) -> None:
        """Upsert the system prompt into the start of the chat session."""
        system_chat = SystemChat(
            content=prompt,
            role=ChatRole.SYSTEM,
            id=uuid4(),
            render_chat=False,
        )
        self.chats.insert(0, system_chat)

    def remove_system_prompt(self) -> None:
        """Remove the system prompt from the chat session."""
        if self.chats and isinstance(self.chats[0], SystemChat):
            self.chats.pop(0)
