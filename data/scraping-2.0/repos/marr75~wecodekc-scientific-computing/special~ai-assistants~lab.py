import abc
import inspect
import json
import re
from typing import (
    Type,
    Callable,
    Any,
    get_type_hints,
    Optional,
    cast,
    Mapping,
    Literal,
    TypeAlias,
    Iterable,
)

import dotenv

dotenv.load_dotenv()

import openai
from mypy_extensions import KwArg
from openai.types import FunctionDefinition
from openai.types.beta.thread import Thread
from openai.types.beta.threads import Run, ThreadMessage
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.beta.assistant_create_params import Tool as ToolAssistantTool
from openai.types.beta.assistant import Assistant, Tool, ToolFunction

import pydantic
import tenacity


class AutoNamer:
    """
    Mixin class that provides a default name and description based on the class name and docstring.
    """

    @property
    def name(self) -> str:
        """
        The name of the class with spaces between camel casing.
        Ex: FunctionTool -> Function Tool
        """
        name = type(self).__name__
        # Add spaces between camel casing
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        return name

    @property
    def description(self) -> str:
        """
        The docstring of the class, with all whitespace collapsed to a single space.
        """
        doc = type(self).__doc__
        assert doc is not None, "FunctionTool subclasses must have a docstring"
        doc = re.sub(r"\s+", " ", doc.strip())
        return doc


class FunctionTool(AutoNamer, ToolFunction):
    """
    A base class, mixin, or concrete class (when a callback is provided) for defining Functions the AI Assistant can call.
    """

    parameter_type: Type[pydantic.BaseModel]
    callback: Optional[Callable[[KwArg(Any)], str]]

    def run(self, **kwargs: Any) -> str:
        """
        The function to be called by the AI Assistant. Must be implemented by subclasses or provided as a callback.
        """
        if self.callback is not None:
            return self.callback(**kwargs)
        else:
            raise NotImplementedError(
                "FunctionTool subclasses must implement run() or provide callback"
            )

    def __init__(self, callback: Optional[Callable[[KwArg(Any)], str]] = None) -> None:
        """
        Initializes the FunctionTool.
        callback - an optional callback function to be invoked instead of run()
        """
        # Inspect the callback or run method to generate parameter types.
        method_to_inspect = callback or self.run
        signature = inspect.signature(method_to_inspect)
        type_hints = get_type_hints(method_to_inspect)
        fields = {
            k: (v, ...) for k, v in type_hints.items() if k in signature.parameters
        }
        parameter_type = pydantic.create_model(
            "ParameterModel",
            **fields,
        )
        definition = FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=parameter_type.model_json_schema(),
        )
        super().__init__(
            function=definition,
            type="function",
            callback=callback,
            parameter_type=parameter_type,
        )


class AssistantManager(AutoNamer, abc.ABC):
    """
    A base class for managing an AI Assistant.
    """

    model: Literal["gpt-3.5-turbo-1106"] = "gpt-3.5-turbo-1106"

    @property
    def instructions(self) -> str:
        """
        The instructions for the AI Assistant.
        """
        return self.description

    @property
    def tool_map(self) -> Mapping[str, Callable[[KwArg(Any)], str]]:
        """
        A mapping of tool names to the functions that should be called when the tool is invoked.
        """
        mapping = {
            tool.name: tool.run for tool in self.tools if isinstance(tool, FunctionTool)
        }
        return mapping

    @property
    def tool_meta(self) -> list[ToolAssistantTool]:
        """
        A list of tool metadata to be passed to the API when creating the assistant.
        """
        meta = [
            cast(ToolAssistantTool, tool.model_dump(include={"type", "function"}))
            for tool in self.tools
        ]
        return meta

    @tenacity.retry(
        retry=tenacity.retry_if_result(
            lambda run: run.status in ("queued", "in_progress")
        ),
        # Exponential backoff starting at 1s and maxing at 60s
        wait=tenacity.wait.wait_exponential(min=0.1, multiplier=2, max=10),
        stop=tenacity.stop.stop_after_attempt(10),
    )
    def _wait_for_run(
        self, run: Optional[Run] = None, run_id: Optional[str] = None
    ) -> Run:
        """
        Waits for the run to complete.
        """
        if (run is None) == (run_id is None):  # Ensure one and only one is provided
            raise ValueError("Must provide exactly one of run or run_id")
        elif run is not None:
            run_id = run.id
        else:
            run_id = cast(str, run_id)
        run = openai.beta.threads.runs.retrieve(
            thread_id=self.thread.id,
            run_id=run_id,
        )
        return run

    def run_thread(self, run: Optional[Run] = None) -> Run:
        """
        Runs the thread until completion. Responds to required actions (tool dispatches) as they are encountered.
        """
        if run is None:
            run = openai.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant_instance.id,
            )
        while True:
            run = self._wait_for_run(run)
            if run.status == "requires_action":
                self.dispatch(run)
            elif run.status == "completed":
                return run
            else:
                raise RuntimeError(
                    f"Run {run.id} failed with status {run.status} while processing thread {self.thread.id}."
                )

    def dispatch(self, run: Run) -> None:
        """
        Dispatches the required action for the run. Currently only supports tool dispatches.
        """
        assert run.required_action is not None, "Run requires action but has none"
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = [
            cast(
                ToolOutput,
                {
                    "output": self.tool_map[tool_call.function.name](
                        **json.loads(tool_call.function.arguments)
                    ),
                    "tool_call_id": tool_call.id,
                },
            )
            for tool_call in tool_calls
        ]
        openai.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs,
        )

    def add_message(self, content: str, run: bool = True) -> list[ThreadMessage]:
        """
        Adds a message to the thread, optionally running the thread after the message is added.
        """
        message = openai.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=content,
        )
        if run:
            _ = self.run_thread()
            new_messages = openai.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="desc",
                before=message.id,
            )
            return [message] + list(new_messages)
        else:
            return [message]

    def create_assistant(self) -> Assistant:
        """
        Creates a new assistant based on the metadata of this class and its tools.
        """
        return openai.beta.assistants.create(
            model=self.model,
            name=self.name,
            instructions=self.instructions,
            tools=self.tool_meta,
        )

    def cleanup(self) -> None:
        """
        Cleans up the assistant and thread.
        """
        openai.beta.threads.delete(thread_id=self.thread.id)
        openai.beta.assistants.delete(assistant_id=self.assistant_instance.id)

    def __init__(
        self,
        tools: Iterable[Tool] = tuple(),
        assistant_instance: Optional[Assistant] = None,
        thread: Optional[Thread] = None,
    ) -> None:
        """
        Initializes the assistant manager.
        tools - an iterable of tools to be added to the assistant
        assistant_instance - an optional assistant instance to use instead of creating a new one
        thread - an optional thread to use instead of creating a new one
        """
        if assistant_instance is None:
            assistant_instance = self.create_assistant()
        self.assistant_instance = assistant_instance
        if thread is None:
            thread = openai.beta.threads.create()
        self.thread = thread
        if not hasattr(self, "tools"):
            self.tools = list(tools)
        else:
            self.tools.extend(tools)


if __name__ == "__main__":
    _operators: TypeAlias = Literal["+", "-", "*", "/", "//", "%", "**"]

    class Calculator(FunctionTool):
        """
        A simple calculator tool
        Supports basic binary operators
        Calls are independent, so you can't do things like calculator(1, 2, +) calculator(3, None, *)
        to calculate (1 + 2) * 3
        You must call calculator(1, 2, +) and then calculator(previous_result, 3, *)
        """

        def run(self, lhs: float, rhs: float, operator: _operators) -> str:
            functional_operator = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "/": lambda a, b: a / b,
                "//": lambda a, b: a // b,
                "%": lambda a, b: a % b,
                "**": lambda a, b: a**b,
            }[operator]
            return str(functional_operator(lhs, rhs))

    class CalculatorAssistant(AssistantManager):
        """Calculate math expressions for the user"""

        tools = [
            Calculator(),
        ]

    manager = CalculatorAssistant()
