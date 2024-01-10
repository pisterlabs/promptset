from typing import (
    Callable,
    Dict,
    Generator,
    AsyncGenerator,
    List,
    Literal,
    Sequence,
    Tuple,
    TypeVar,
    Any,
    Generic,
    overload,
    TYPE_CHECKING,
)

from .message import *
from dotenv import dotenv_values
import inspect
from inspect import Parameter
import openai
import logging
import os
from openai.types.chat import ChatCompletionMessageParam
import asyncio
from datetime import datetime

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plugins import Plugin


@dataclass
class GPTOptions:
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    stop: Optional[str] | List[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    timeout: Optional[float] = None

    def as_kwargs(self) -> dict[str, Any]:
        args = {
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }
        return {k: v for k, v in args.items() if v is not None}


M = TypeVar("M", Message, Message | MessageStream)


class ChatCompletion(Generic[M]):
    def __init__(self, agen: AsyncGenerator[M, None]) -> None:
        super().__init__()
        self.__agen = agen

    async def __anext__(self) -> M:
        return await self.__agen.__anext__()

    def __aiter__(self):
        return self

    def __next__(self) -> M:
        loop = asyncio.get_event_loop()
        try:
            result = loop.run_until_complete(self.__anext__())
            return result
        except StopAsyncIteration:
            raise StopIteration()

    def __iter__(self):
        return self


class AugmentedGPT:
    def support_tools(self) -> bool:
        return self.model in [
            "gpt-4-1106-preview",
            "gpt-3.5-turbo-1106",
        ]

    def __init__(
        self,
        model: str
        | Literal[
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            # Preview versions
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
            "gpt-3.5-turbo-1106",
        ] = "gpt-4-1106-preview",
        functions: Optional[Sequence[Callable[..., Any]]] = None,
        plugins: Optional[Sequence["Plugin"]] = None,
        debug: bool = False,
        gpt_options: Optional[GPTOptions] = None,
        api_key: Optional[str] = None,
        prologue: Optional[List[Message]] = None,
        inject_current_date_time: bool = False,
    ):
        self.gpt_options = gpt_options or GPTOptions()
        self.model = model
        api_key = api_key or dotenv_values().get(
            "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")
        )
        assert api_key is not None, "Missing OPENAI_API_KEY"
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.logger = logging.getLogger("AugmentedGPT")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.__functions: Dict[str, Tuple[Any, Callable[..., Any]]] = {}
        for f in functions or []:
            self.add_function(f)
        self.__plugins: Any = {}
        for p in plugins or []:
            clsname = p.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            p.register(self)
            self.__plugins[clsname] = p
        self.__prologue = prologue or []
        self.history: List[Message] = [m for m in self.__prologue] or []
        self.inject_current_date_time = inject_current_date_time

    def reset(self):
        self.history = [m for m in self.__prologue]

    def get_plugin(self, name: str) -> "Plugin":
        return self.__plugins[name]

    def add_function(self, f: Callable[..., Any]):
        func_info = getattr(f, "gpt_function_call_info")
        self.logger.info("Register-Function: " + func_info["name"])
        self.__functions[func_info["name"]] = (func_info, f)

    def __filter_args(self, callable: Callable[..., Any], args: Any):
        p_args: List[Any] = []
        kw_args: Dict[str, Any] = {}
        for p in inspect.signature(callable).parameters.values():
            match p.kind:
                case Parameter.POSITIONAL_ONLY:
                    p_args.append(args[p.name] if p.name in args else p.default.default)
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                    kw_args[p.name] = (
                        args[p.name] if p.name in args else p.default.default
                    )
                case other:
                    raise ValueError(f"{other} is not supported")
        return p_args, kw_args

    async def __call_function(
        self, function_call: FunctionCall, tool_id: Optional[str]
    ) -> Message:
        func_name = function_call.name
        key = func_name if not func_name.startswith("functions.") else func_name[10:]
        if tool_id is not None:
            result_msg = Message(role=Role.TOOL, tool_call_id=tool_id, content="")
        else:
            result_msg = Message(role=Role.FUNCTION, name=func_name, content="")
        try:
            func = self.__functions[key][1]
            arguments = function_call.arguments
            args, kw_args = self.__filter_args(func, arguments)
            self.logger.debug(
                f"➡️ {func_name}: "
                + ", ".join(str(a) for a in args)
                + ", ".join((f"{k}={v}" for k, v in kw_args.items()))
            )
            result_or_coroutine = func(*args, **kw_args)
            if inspect.iscoroutine(result_or_coroutine):
                result = await result_or_coroutine
            else:
                result = result_or_coroutine
            if not isinstance(result, str):
                result = json.dumps(result)
            result_msg.content = result
        except Exception as e:
            print(e)
            result_msg.content = f"Failed to execute function `{func_name}`. Please retry. Error Message: {e}"
        return result_msg

    @overload
    async def __chat_completion_request(
        self, messages: List[Message], stream: Literal[False]
    ) -> Message:
        ...

    @overload
    async def __chat_completion_request(
        self, messages: List[Message], stream: Literal[True]
    ) -> MessageStream:
        ...

    async def __chat_completion_request(
        self, messages: List[Message], stream: bool
    ) -> Message | MessageStream:
        functions = [x for (x, _) in self.__functions.values()]
        msgs: List[ChatCompletionMessageParam] = [
            m.to_chat_completion_message_param() for m in messages
        ]
        args: Any = {
            "model": self.model,
            "messages": msgs,
            **self.gpt_options.as_kwargs(),
        }
        if len(functions) > 0:
            if self.support_tools():
                args["tools"] = [{"type": "function", "function": f} for f in functions]
                args["tool_choice"] = "auto"
            else:
                args["functions"] = functions
                args["function_call"] = "auto"
        if stream:
            response = await self.client.chat.completions.create(**args, stream=True)
            return MessageStream(response)
        else:
            response = await self.client.chat.completions.create(**args, stream=False)
            return Message.from_chat_completion_message(response.choices[0].message)

    @overload
    async def __chat_completion(
        self,
        messages: List[Message],
        stream: Literal[False] = False,
        context_free: bool = False,
    ) -> Generator[Message, None, None]:
        ...

    @overload
    async def __chat_completion(
        self, messages: List[Message], stream: Literal[True], context_free: bool = False
    ) -> Generator[Message | MessageStream, None, None]:
        ...

    async def __chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context_free: bool = False,
    ):
        history = [h for h in (self.history if not context_free else [])]
        old_history_length = len(history)
        if self.inject_current_date_time:
            dt = datetime.today().strftime("%Y-%m-%d %a %H:%M:%S")
            history.append(Message(role=Role.SYSTEM, content=f"current-time: {dt}"))
        history.extend(messages)
        for m in messages:
            await self.__on_new_chat_message(m)
        # First completion request
        message: Message
        if stream:
            s = await self.__chat_completion_request(history, stream=True)
            yield s
            message = s.message()
        else:
            message = await self.__chat_completion_request(history, stream=False)
            yield message
        history.append(message)
        await self.__on_new_chat_message(message)
        while message.function_call is not None or len(message.tool_calls) > 0:
            if len(message.tool_calls) > 0:
                for t in message.tool_calls:
                    assert t.type == "function"
                    result = await self.__call_function(t.function, tool_id=t.id)
                    history.append(result)
                    await self.__on_new_chat_message(result)
                    yield result
            else:
                assert message.function_call is not None
                # ChatGPT wanted to call a user-defined function
                result = await self.__call_function(message.function_call, tool_id=None)
                history.append(result)
                await self.__on_new_chat_message(result)
                yield result
            # Send back the function call result
            message: Message
            if stream:
                r = await self.__chat_completion_request(history, stream=True)
                yield r
                message = r.message()
            else:
                message = await self.__chat_completion_request(history, stream=False)
                yield message
            history.append(message)
            await self.__on_new_chat_message(message)
        if not context_free:
            self.history.extend(history[old_history_length:])

    @overload
    def chat_completion(
        self,
        messages: List[Message],
        stream: Literal[False] = False,
        context_free: bool = False,
    ) -> ChatCompletion[Message]:
        ...

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[True], context_free: bool = False
    ) -> ChatCompletion[Message | MessageStream]:
        ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context_free: bool = False,
    ):
        if stream:
            return ChatCompletion(
                self.__chat_completion(messages, stream=True, context_free=context_free)
            )
        else:
            return ChatCompletion(
                self.__chat_completion(
                    messages, stream=False, context_free=context_free
                )
            )

    async def __on_new_chat_message(self, msg: Message):
        for p in self.__plugins.values():
            result = p.on_new_chat_message(msg)
            if inspect.iscoroutine(result):
                await result
