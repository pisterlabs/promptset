from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union
from human_id import generate_id

from pydantic import Field

from langchain.tools.base import BaseTool


class InteractionType(Enum):
    User = "user"
    Agent = "agent"
    Tool = "tool"


class StopMotive(Enum):
    """The reason an agent stopped."""

    Finished = "finished"
    """The agent finished its work."""
    Error = "error"
    """The agent encountered an error."""


@dataclass(order=True)
class Interaction:
    priority: int
    interaction_type: InteractionType = field(compare=False)
    who: str = field(compare=False)
    inputs: dict[str, Any] = field(compare=False)


@dataclass
class AgentActionWithId:
    """A full description of an action for an ActionAgent to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action."""

    # NOTE: field with factory is disabled as the object would then be not serializable
    job_id: str = field(default_factory=generate_id)
    """The human-readable ID of the job that this action is a part of."""


class BaseParallelizableTool(BaseTool):
    """Base class for tools that can be run in parallel with other tools."""

    is_parallelizable: bool = Field(
        default=False,
        const=True,
        description="Whether this tool can be run in parallel with other tools.",
    )

    global_context: Any
    context: Any

    def _set_context(self, **kwargs) -> None:
        """Sets the context of the tool."""

        if self.context is None:
            self.context = {}

        self.context.update(kwargs)

    def invoke(
        self,
        global_context: dict[str, Any],
        context: dict[str, Any],
        *args,
        **kwargs,
    ) -> Any:
        """Invokes the tool."""
        self.global_context = global_context
        self._set_context(**context)
        return self.run(*args, **kwargs)

    def _arun(self, *args: Any, **kwargs: Any):
        return self._run(*args, **kwargs)
