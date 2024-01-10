from typing import Generator
from json import loads

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from openai.types.chat import ChatCompletion
from openai._streaming import Stream
from zkappumstad.runners import RunnerMessage, ToolMessage, StateChange, StreamMessage

from zkappumstad.tools import (
    Tool,
    doc_tool,
    code_tool,
    project_tool,
    issue_tool,
    state_change_tool,
)
from zkappumstad.prompt import SYSTEM_PROMPT

load_dotenv(find_dotenv(".env.local"), override=True)

client = OpenAI()
tools: dict[str, Tool] = {
    tool.name: tool
    for tool in [doc_tool, code_tool, project_tool, issue_tool, state_change_tool]
}


def create_completion(history, message) -> Generator[RunnerMessage, None, None]:
    while True:
        try:
            history.append({"role": "user", "content": message})
            chat_completion: Stream[ChatCompletion] = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,
                ],
                model="gpt-4-1106-preview",
                temperature=0.2,
                stream=True,
                functions=[tool.description for tool in tools.values()],
                function_call="auto",
            )

            if not isinstance(chat_completion, Stream):
                continue

            part = next(chat_completion)

            if part.choices[0].delta.function_call:
                function_name = part.choices[0].delta.function_call.name
                tool = tools[function_name]

                yield ToolMessage(tool.message, "TOOL_MESSAGE")
                args = "".join(
                    list(
                        part.choices[0].delta.function_call.arguments
                        for part in chat_completion
                        if part.choices[0].delta.function_call
                    )
                )
                history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": function_name, "arguments": args},
                    }
                )

                args = loads(args)
                result = tool.function(**args)
                if isinstance(result, StateChange):
                    yield result
                    break
                history.append(
                    {"role": "function", "name": function_name, "content": result}
                )

                continue

            yield StreamMessage(part.choices[0].delta.content or "", "STREAM_MESSAGE")
            for part in chat_completion:
                yield StreamMessage(
                    part.choices[0].delta.content or "", "STREAM_MESSAGE"
                )
            break

        except Exception as e:
            print(e)
            print(e.with_traceback())
            return None


if __name__ == "__main__":
    create_completion([], "What's the weather in San Francisco?")
