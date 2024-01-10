import json
import os
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..schema import BaseMessage, FunctionAvailable, FunctionCall, Role, SystemMessage


if TYPE_CHECKING:
    from openai import Stream
    from openai.types.chat import ChatCompletion, ChatCompletionChunk


class ChatOpenAI:
    def __init__(self) -> None:
        self.model = os.environ.get("CHAT_MODEL")
        self._client = OpenAI(max_retries=5, timeout=30.0)

    def _parse_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def _parse_tools(self, tools: List[FunctionAvailable]) -> List[Dict[str, Any]]:
        return [{"type": tool.type, "function": tool.function} for tool in tools]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _completion_with_backoff(
        self,
        messages: List[BaseMessage],
        stream: Optional[bool] = False,
        tools: Optional[List[FunctionAvailable]] = None,
    ) -> Union["ChatCompletion", "Stream[ChatCompletionChunk]"]:
        if messages[0].role != Role.SYSTEM and os.environ.get("DEFAULT_SYSTEM_PROMPT"):
            messages.insert(0, SystemMessage(content=os.environ.get("DEFAULT_SYSTEM_PROMPT")))

        request_kwargs = {"messages": self._parse_messages(messages), "model": self.model, "stream": stream}
        if tools is not None:
            request_kwargs["tools"] = self._parse_tools(tools)

        if os.environ.get("TEMPERATURE"):
            request_kwargs["temperature"] = float(os.environ.get("TEMPERATURE"))

        return self._client.chat.completions.create(**request_kwargs)

    def chat(self, messages: List[BaseMessage]) -> str:
        return self._completion_with_backoff(messages=messages).choices[0].message.content

    def stream_chat(self, messages: List[BaseMessage]) -> Generator[str, None, None]:
        for chunk in self._completion_with_backoff(messages=messages, stream=True):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def function_call(self, messages: List[BaseMessage], tools: List[FunctionAvailable]) -> FunctionCall:
        tool_call = (
            self._completion_with_backoff(messages=messages, tools=tools).choices[0].message.tool_calls[0]
        )  # current only support a single tool
        return FunctionCall(name=tool_call.function.name, arguments=json.loads(tool_call.function.arguments))


if __name__ == "__main__":
    from ..schema import HumanMessage

    chat_openai = ChatOpenAI()
    messages = [HumanMessage(content="Say this is a test")]
    print(chat_openai.chat(messages))
