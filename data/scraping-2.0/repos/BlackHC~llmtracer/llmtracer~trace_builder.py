"""
Simple logger/execution tracker that uses tracks the stack frames and 'data'.
"""
#  LLM Tracer
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import inspect
import time
import traceback
import typing
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import ClassVar

from langchain.schema import BaseMessage

from llmtracer import module_filtering
from llmtracer.frame_info import FrameInfo, get_frame_infos
from llmtracer.object_converter import DynamicObjectConverter, ObjectConverter, convert_pydantic_model
from llmtracer.trace_schema import Trace, TraceNode, TraceNodeKind
from llmtracer.utils.callable_wrapper import CallableWrapper
from llmtracer.utils.weakrefs import WeakKeyIdMap

T = typing.TypeVar("T")
P = typing.ParamSpec("P")


trace_object_converter = DynamicObjectConverter()
trace_module_filters = None

# TODO: move this somewhere else?
# chat messages need to be converted to JSON
trace_object_converter.register_converter(convert_pydantic_model, BaseMessage)


def default_timer() -> int:
    """
    Default timer for the tracer.

    Returns:
        The current time in milliseconds.
    """
    return int(time.time() * 1000)


@dataclass
class TraceNodeBuilder:
    """
    A node builder in the trace tree.
    """

    kind: TraceNodeKind
    name: str | None
    event_id: int
    start_time_ms: int
    delta_frame_infos: list[FrameInfo]
    stack_height: int

    end_time_ms: int | None = None
    parent: 'TraceNodeBuilder | None' = None
    children: list['TraceNodeBuilder'] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create_root(cls):
        return cls(
            kind=TraceNodeKind.SCOPE,
            name=None,
            event_id=0,
            start_time_ms=0,
            delta_frame_infos=[],
            stack_height=0,
        )

    def get_delta_frame_infos(
        self, num_frames_to_skip: int = 0, module_filters: module_filtering.ModuleFilters | None = None, context=3
    ):
        frame_infos, full_stack_height = get_frame_infos(
            num_top_frames_to_skip=num_frames_to_skip + 1,
            num_bottom_frames_to_skip=self.stack_height,
            module_filters=module_filters,
            context=context,
        )

        return frame_infos, full_stack_height

    def build(self):
        return TraceNode(
            kind=self.kind,
            name=self.name,
            event_id=self.event_id,
            start_time_ms=self.start_time_ms,
            end_time_ms=self.end_time_ms or default_timer(),
            running=self.end_time_ms is None,
            delta_frame_infos=self.delta_frame_infos,
            properties=self.properties,
            children=[sub_event.build() for sub_event in self.children],
        )


class TraceBuilderEventHandler:
    def on_scope_final(self, builder: 'TraceBuilder'):
        pass

    def on_event_scope_final(self, builder: 'TraceBuilder'):
        pass


@dataclass(weakref_slot=True, slots=True)
class TraceBuilder:
    _current: ClassVar[ContextVar['TraceBuilder | None']] = ContextVar("current_trace_builder", default=None)

    module_filters: module_filtering.ModuleFilters
    stack_frame_context: int

    event_root: TraceNodeBuilder = field(default_factory=TraceNodeBuilder.create_root)
    object_map: WeakKeyIdMap[object, str] = field(default_factory=WeakKeyIdMap)
    unique_objects: dict[str, dict] = field(default_factory=dict)

    id_counter: int = 0
    current_event_node: TraceNodeBuilder | None = None

    event_handlers: list[TraceBuilderEventHandler] = field(default_factory=list)

    def build(self):
        return Trace(
            name=self.event_root.name,
            properties=self.event_root.properties,
            traces=[child.build() for child in self.event_root.children],
            unique_objects=self.unique_objects,
        )

    def next_id(self):
        self.id_counter += 1
        return self.id_counter

    @contextmanager
    def scope(self, name: str | None = None):
        """
        Context manager that allows to trace our program execution.
        """
        assert self.current_event_node is None
        self.current_event_node = self.event_root

        token = self._current.set(self)
        try:
            with self.event_scope(name=name, kind=TraceNodeKind.SCOPE, skip_frames=2):
                yield self
        finally:
            for handler in self.event_handlers:
                handler.on_scope_final(self)
            self._current.reset(token)
            self.current_event_node = None

    @contextmanager
    def event_scope(
        self,
        name: str | None,
        properties: dict[str, object] | None = None,
        kind: TraceNodeKind = TraceNodeKind.SCOPE,
        skip_frames: int = 0,
    ):
        """
        Context manager that allows to trace our program execution.
        """
        assert self._current.get() is self
        assert self.current_event_node is not None

        if properties is None:
            properties = {}

        start_time = default_timer()
        delta_frame_infos, stack_height = self.current_event_node.get_delta_frame_infos(
            num_frames_to_skip=2 + skip_frames, module_filters=self.module_filters, context=self.stack_frame_context
        )
        event_node = TraceNodeBuilder(
            kind=kind,
            name=name,
            event_id=self.next_id(),
            start_time_ms=start_time,
            delta_frame_infos=delta_frame_infos,
            stack_height=stack_height - 1,
            parent=self.current_event_node,
            properties=dict(properties),
        )
        self.current_event_node.children.append(event_node)

        old_event_node = self.current_event_node
        self.current_event_node = event_node

        try:
            yield
        except BaseException as e:
            self.update_event_properties(exception='\n'.join(traceback.TracebackException.from_exception(e).format()))
            raise
        finally:
            event_node.end_time_ms = default_timer()
            self.current_event_node = old_event_node

            for handler in self.event_handlers:
                handler.on_event_scope_final(self)

    def register_object(self, obj: object, name: str, properties: dict[str, object]):
        # Make name unique if needed
        if name in self.unique_objects:
            # if we are in a scope, we can use the scope name as a prefix
            if self.current_event_node is not None:
                name = f"{self.current_event_node.name}_{name}"

            if name in self.unique_objects:
                i = 1
                while f"{name}[{i}]" in self.unique_objects:
                    i += 1
                name = f"{name}[{i}]"
        self.object_map[obj] = name
        self.unique_objects[name] = properties

    def convert_object(self, obj: object, preferred_object_converter: ObjectConverter | None = None):
        if preferred_object_converter is None:
            preferred_object_converter = self.convert_object

        # if the object is in the map, we return its name as a reference
        if obj in self.object_map:
            return dict(unique_object=self.object_map[obj])

        return trace_object_converter(obj, preferred_object_converter)

    @classmethod
    def get_current(cls) -> 'TraceBuilder | None':
        return cls._current.get()

    @classmethod
    def get_current_node(cls) -> 'TraceNodeBuilder | None':
        current = cls.get_current()
        if current is None:
            return None
        else:
            return current.current_event_node

    def add_event(
        self,
        name: str,
        properties: dict[str, object] | None = None,
        kind: TraceNodeKind = TraceNodeKind.EVENT,
    ):
        """
        Add an event to the current scope.
        """
        if properties is None:
            properties = {}
        with self.event_scope(name, properties=properties, kind=kind, skip_frames=2):
            pass

    def update_event_properties(self, properties: dict[str, object] | None = None, /, **kwargs):
        """
        Update the properties of the current event.
        """
        assert self.current_event_node is not None
        if properties is None:
            properties = {}
        self.current_event_node.properties.update(self.convert_object(properties | kwargs))

    def update_name(self, name: str):
        """
        Update the name of the current event.
        """
        assert self.current_event_node is not None
        self.current_event_node.name = name


@dataclass
class CallTracer(CallableWrapper, typing.Callable[P, T], typing.Generic[P, T]):  # type: ignore
    __signature__: inspect.Signature
    __wrapped__: typing.Callable[P, T]
    __wrapped_name__: str
    __kind__: TraceNodeKind = TraceNodeKind.CALL
    __capture_return__: bool = False
    __capture_args__: bool | list[str] | slice = False
    __object_converter__: DynamicObjectConverter | None = None

    def __call__(self, *args, **kwargs):
        # check if we are in a trace
        builder = TraceBuilder.get_current()
        if builder is None:
            return self.__wrapped__(*args, **kwargs)

        object_converter = self.__object_converter__
        if object_converter is None:
            object_converter = builder.convert_object

        # build properties
        properties = {}
        if self.__capture_args__ is not False:
            # bind the arguments to the signature
            bound_args = self.__signature__.bind(*args, **kwargs)
            # add the arguments to the properties
            if self.__capture_args__ is True:
                arguments = bound_args.arguments
            elif isinstance(self.__capture_args__, list):
                arguments = {arg: bound_args.arguments[arg] for arg in self.__capture_args__}
            elif isinstance(self.__capture_args__, slice):
                arguments = {
                    arg: bound_args.arguments[arg] for arg in list(bound_args.arguments)[self.__capture_args__]
                }

            # anything that can be stored in a json is okay
            converted_arguments = {}
            for arg, value in arguments.items():
                converted_arguments[arg] = object_converter(value)
            properties["arguments"] = converted_arguments

        # create event scope
        with builder.event_scope(self.__wrapped_name__, properties, kind=self.__kind__, skip_frames=1):
            # call the function
            result = self.__wrapped__(*args, **kwargs)

            if self.__capture_return__:
                builder.current_event_node.properties.update({"result": object_converter(result)})
        return result


class Slicer:
    def __class_getitem__(cls, item):
        return item


slicer = Slicer


def trace_calls(
    func=None,
    *,
    name: str | None = None,
    kind: TraceNodeKind = TraceNodeKind.CALL,
    capture_return: bool = False,
    capture_args: bool | list[str] | slice = False,
    object_converter: DynamicObjectConverter | None = None,
):
    """
    Decorator that allows to trace our program execution.
    """
    if func is None:
        return partial(
            trace_calls,
            name=name,
            kind=kind,
            capture_return=capture_return,
            capture_args=capture_args,
        )

    # get the signature of the function
    signature = inspect.signature(func)

    # if capture_args is an iterable, convert it to a set
    if isinstance(capture_args, typing.Iterable):
        assert not isinstance(capture_args, bool)
        arg_names = set(capture_args)

        # check that all the arguments are valid
        for arg in arg_names:
            if arg not in signature.parameters:
                raise ValueError(f"Argument '{arg}' is not a valid argument of function '{func.__name__}'!")

    # get the name of the function
    if name is None:
        name = func.__name__

    wrapped_function = wraps(func)(
        CallTracer(
            __signature__=signature,
            __wrapped__=func,
            __wrapped_name__=name,
            __kind__=kind,
            __capture_return__=capture_return,
            __capture_args__=capture_args,
            __object_converter__=object_converter,
        )
    )

    return wrapped_function
