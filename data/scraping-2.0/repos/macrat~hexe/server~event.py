import json
import re
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Literal, TypedDict

from openaitypes import Message as OpenAIMessage

EventType = Literal[
    "user", "assistant", "function_call", "function_output", "status", "error"
]


class EventDictRequired(TypedDict):
    created_at: float
    id: str
    type: EventType


class EventDict(EventDictRequired, total=False):
    arguments: str
    content: str
    delta: Literal[True]
    generating: bool
    name: str
    source: str


@dataclass(init=False)
class Event:
    id: uuid.UUID
    created_at: datetime
    delta: bool

    def __init__(
        self,
        id: uuid.UUID | None = None,
        delta: bool = False,
        created_at: datetime | None = None,
    ) -> None:
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id

        if created_at is None:
            self.created_at = datetime.now(timezone.utc)
        else:
            self.created_at = created_at

        self.delta = delta

    @property
    @abstractmethod
    def type(self) -> EventType:
        ...

    def as_dict(self) -> EventDict:
        ev: EventDict = {
            "type": self.type,
            "id": str(self.id),
            "created_at": self.created_at.timestamp(),
        }
        if self.delta:
            ev["delta"] = True
        return ev

    def as_json(self) -> str:
        return json.dumps(self.as_dict())


@dataclass(init=False)
class User(Event):
    content: str
    type: Literal["user"] = "user"

    def __init__(self, *args, content: str = "", **kwargs) -> None:
        self.content = content
        super().__init__(*args, **kwargs)

    def as_dict(self) -> EventDict:
        return {
            **super().as_dict(),
            "content": self.content,
        }


@dataclass(init=False)
class Assistant(Event):
    content: str
    source: uuid.UUID
    type: Literal["assistant"] = "assistant"

    def __init__(self, content: str, source: uuid.UUID, *args, **kwargs) -> None:
        self.content = content
        self.source = source
        super().__init__(*args, **kwargs)

    def as_dict(self) -> EventDict:
        return {
            **super().as_dict(),
            "content": self.content,
            "source": str(self.source),
        }


@dataclass(init=False)
class FunctionCall(Event):
    name: str
    arguments: str
    source: uuid.UUID
    type: Literal["function_call"] = "function_call"

    def __init__(
        self, name: str, arguments: str, source: uuid.UUID, *args, **kwargs
    ) -> None:
        self.name = name
        self.arguments = arguments
        self.source = source
        super().__init__(*args, **kwargs)

    def as_dict(self) -> EventDict:
        return {
            **super().as_dict(),
            "name": self.name,
            "arguments": self.arguments,
            "source": str(self.source),
        }


@dataclass(init=False)
class FunctionOutput(Event):
    name: str
    content: str
    source: uuid.UUID
    type: Literal["function_output"] = "function_output"

    def __init__(
        self, source: uuid.UUID, name: str, content: str, *args, **kwargs
    ) -> None:
        self.name = name
        self.content = content
        self.source = source
        super().__init__(*args, **kwargs)

    def as_dict(self) -> EventDict:
        return {
            **super().as_dict(),
            "name": self.name,
            "content": self.content,
            "source": str(self.source),
        }

    @property
    def short_content(self) -> str:
        """The content without the URL of media.
        This value is used for the messages to send to LLM.
        """

        m = re.match(
            r'^<(?P<tag>img|video) src="data:(image|video)/[-+_a-zA-Z0-9]+;base64,[^"]+"'
            r' (controls="controls" )?alt="(?P<alt>[^"]+)" />$',
            self.content,
        )
        if m is not None:
            return (
                f'<{m.group("tag")} alt="{m.group("alt")}" src="/*The media has shown,'
                ' but the URL in the chat history has omitted.*/" />'
            )

        return self.content


@dataclass(init=False)
class Status(Event):
    source: uuid.UUID | None
    generating: bool
    type: Literal["status"] = "status"

    def __init__(
        self, *args, source: uuid.UUID | None = None, generating: bool = False, **kwargs
    ) -> None:
        self.source = source
        self.generating = generating

        super().__init__(*args, **kwargs)

    def as_dict(self) -> EventDict:
        return {
            **super().as_dict(),
            "source": str(self.source),
            "generating": self.generating,
        }


@dataclass(init=False)
class Error(Event):
    content: str
    source: uuid.UUID | None
    type: Literal["error"] = "error"

    def __init__(self, source: uuid.UUID, *args, content: str = "", **kwargs) -> None:
        self.content = content
        self.source = source
        super().__init__(*args, **kwargs)
        self.source = source

    def as_dict(self) -> EventDict:
        return {
            **super().as_dict(),
            "content": self.content,
            "source": str(self.source),
        }


def as_messages(events: list[Event]) -> Iterator[OpenAIMessage]:
    skip = False

    for i, event in enumerate(events):
        if skip:
            skip = False
            continue

        next_event = events[i + 1] if i + 1 < len(events) else None

        match event, next_event:
            case User() as ev, _:
                yield {
                    "role": "user",
                    "content": ev.content,
                }
            case Assistant() as ev, FunctionCall() as nxt:
                skip = True
                yield {
                    "role": "assistant",
                    "content": ev.content,
                    "function_call": {
                        "name": nxt.name,
                        "arguments": nxt.arguments,
                    },
                }
            case Assistant() as ev, _:
                yield {
                    "role": "assistant",
                    "content": ev.content,
                }
            case FunctionCall() as ev, _:
                yield {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": ev.name,
                        "arguments": ev.arguments,
                    },
                }
            case FunctionOutput() as ev, _:
                yield {
                    "role": "function",
                    "name": ev.name,
                    "content": ev.short_content,
                }
            case Error() as ev, _:
                yield {
                    "role": "system",
                    "content": f"Error: {ev.content}",
                }
