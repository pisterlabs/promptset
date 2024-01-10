from typing import (
    Dict,
    Union,
    List,
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, ChatMessage
from funcy import omit


OpenAIChatMessage = Union[BaseMessage, Dict[str, str]]


def convert_message_to_dict(message: BaseMessage) -> Dict[str, str]:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def convert_dict_to_message(_dict: Dict[str, str]) -> BaseMessage:
    role = _dict["role"]
    additional_kwargs = dict(omit(_dict, ["role", "content"]))
    if role == "user":
        return HumanMessage(content=_dict["content"], additional_kwargs=additional_kwargs)
    elif role == "assistant":
        return AIMessage(content=_dict["content"], additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"], additional_kwargs=additional_kwargs)
    else:
        return ChatMessage(content=_dict["content"], role=role, additional_kwargs=additional_kwargs)


def prepare_message(message: OpenAIChatMessage) -> BaseMessage:
    if isinstance(message, dict):
        return convert_dict_to_message(message)
    elif isinstance(message, BaseMessage):
        return message
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def prepare_messages(messages: List[OpenAIChatMessage]) -> List[BaseMessage]:
    return [prepare_message(message) for message in messages]

