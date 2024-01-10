import inspect
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
)

from llmtaskgraph.types import JSON, JSONValue, Prompt

from .api_handler import OpenAiChatApiHandler


if TYPE_CHECKING:
    from .task_graph import GraphContext

_api_handler = OpenAiChatApiHandler()

T = TypeVar("T", covariant=True)
P = ParamSpec("P")
Q = Concatenate["GraphContext", P]


class FunctionId(Generic[P, T]):
    def __init__(self, func: Callable[Q[P], T] | Callable[P, Awaitable[T]]):
        self.name: str = func.__name__
        func_return_type: Any | None = inspect.get_annotations(func).get("return")
        self.func_return_type: type[T] = func_return_type if func_return_type else Any

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return self.name == other.name

    def __repr__(self):
        return f"FunctionId({self.name})"

    def to_json(self) -> JSONValue:
        return self.name

    @classmethod
    def from_json(cls, json: JSONValue) -> "FunctionId[..., Any]":
        assert isinstance(json, str)
        function_id: FunctionId[..., Any] = FunctionId(lambda: None)  # type: ignore

        function_id.name = json

        return function_id


def add_context(fn: Callable[P, T]) -> Callable[Q[P], T]:
    def wrapped_fn(_: "GraphContext", *args: P.args, **kwargs: P.kwargs) -> T:
        return fn(*args, **kwargs)

    wrapped_fn.__name__ = fn.__name__
    return wrapped_fn


class FunctionRegistry:
    def __init__(self):
        self._registry: dict[FunctionId[..., Any], Callable[..., Any]] = {}

    def register(self, func: Callable[Q[P], T]) -> FunctionId[P, T]:
        function_id = FunctionId[P, T](func)
        self._registry[function_id] = func
        return function_id

    def register_no_context(self, func: Callable[P, T]) -> FunctionId[P, T]:
        return self.register(add_context(func))

    def register_api_handler(self, func: Callable[P, Awaitable[T]]) -> FunctionId[P, T]:
        function_id = FunctionId[P, T](func)
        self._registry[function_id] = func
        return function_id

    def copy(self) -> "FunctionRegistry":
        copy: FunctionRegistry = FunctionRegistry()
        copy._registry = self._registry.copy()
        return copy

    def merge(self, other: "FunctionRegistry") -> "FunctionRegistry":
        merged: FunctionRegistry = self.copy()
        merged._registry.update(other._registry)
        return merged

    def get_api_handler(
        self, function_id: FunctionId[P, T]
    ) -> Callable[P, Awaitable[T]]:
        return self._registry[function_id]  # type: ignore

    def __getitem__(self, function_id: FunctionId[P, T]) -> Callable[Q[P], T]:
        return self._registry[function_id]  # type: ignore


def _dont_parse(x: str) -> str:
    return x


def _parse_json(json_str: str) -> JSON:
    return json.loads(json_str)


def _forward_graph_input(context: "GraphContext") -> JSONValue:
    return context.graph_input()


_base_registry = FunctionRegistry()
openai_chat: FunctionId[[Prompt, JSON], str] = _base_registry.register_api_handler(
    _api_handler.api_call
)
dont_parse: FunctionId[[str], str] = _base_registry.register_no_context(_dont_parse)
parse_json: FunctionId[[str], JSON] = _base_registry.register_no_context(_parse_json)
forward_graph_input: FunctionId[[], JSONValue] = _base_registry.register(
    _forward_graph_input
)


def make_base_registry() -> FunctionRegistry:
    return _base_registry.copy()
