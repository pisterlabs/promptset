"""A proxy for the OpenAI API."""
import asyncio
import json
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, UploadFile
from openai import AsyncOpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_create_params import (
    Tool,
    ToolAssistantToolsCode,
    ToolAssistantToolsFunction,
    ToolAssistantToolsRetrieval,
)
from openai.types.beta.thread import Thread
from openai.types.beta.thread_create_and_run_params import (
    ThreadMessage as ThreadMessageCreate,
)
from openai.types.beta.threads.required_action_function_tool_call import (
    RequiredActionFunctionToolCall,
)
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.beta.threads.runs.code_tool_call import CodeToolCall
from openai.types.beta.threads.runs.message_creation_step_details import (
    MessageCreationStepDetails,
)
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.thread_message import ThreadMessage
from openai.types.file_object import FileObject
from sse_starlette.sse import EventSourceResponse

from .lib.decorators import handle, setup_logging
from .lib.functions import use_image, use_instruction
from .lib.schemas import IAssistant
from .lib.services import OpenAIFunction, Stack
from .lib.vector import QueryBuilder as Builder
from .lib.vector import VectorClient as VectorStore

logger = setup_logging(__name__)
ai = AsyncOpenAI()
client = VectorStore()
Result = object
builder = Builder()


@handle
async def similarity_search(text: str, namespace: str) -> str:
    embedding = await ai.embeddings.create(input=text, model="text-embedding-ada-002")
    vector = embedding.data[0].embedding
    response = await client.query(
        expr=(builder("namespace") == namespace).query, vector=vector
    )
    ctx = "\n".join(response)
    return f"Similar results from the knowledge base:\n\n{ctx}"


@handle
async def upsert_vector(text: str, namespace: str):
    embedding = await ai.embeddings.create(input=text, model="text-embedding-ada-002")
    vector = embedding.data[0].embedding
    return await client.upsert(
        vectors=[vector], metadata=[{"namespace": namespace, "text": text}]
    )


@handle
async def function_call(*, name: str, arguments: str, id: str) -> ToolOutput:
    for subclass in OpenAIFunction.__subclasses__():
        if subclass.__name__ == name:
            instance = subclass.parse_raw(arguments)  # type: ignore
            obj = await instance()
            if hasattr(obj, "json") and callable(obj.json):
                data = obj.json()
                if isinstance(data, dict):
                    output = json.dumps(data)
                else:
                    output = data
                return {"output": output, "tool_call_id": id}
            if hasattr(obj, "json") and not callable(obj.json):
                data = obj.json
                if isinstance(data, dict):
                    output = json.dumps(data)
                else:
                    output = data
                return {"output": output, "tool_call_id": id}
            if hasattr(obj, "dict") and callable(obj.dict):
                data = obj.dict()
                if isinstance(data, dict):
                    output = json.dumps(data)
                else:
                    output = data
                return {"output": output, "tool_call_id": id}
            if hasattr(obj, "dict") and not callable(obj.dict):
                data = obj.dict
                if isinstance(data, dict):
                    output = json.dumps(data)
                else:
                    output = data
                return {"output": output, "tool_call_id": id}
            if isinstance(obj, dict):
                output = json.dumps(obj)
                return {"output": output, "tool_call_id": id}
            return {"output": str(obj), "tool_call_id": id}
    raise ValueError(f"Function {name} not found")


@handle
async def exec_function_call(*, call: RequiredActionFunctionToolCall) -> ToolOutput:
    name = call.function.name
    arguments = call.function.arguments
    id = call.id
    return await function_call(name=name, arguments=arguments, id=id)


@handle
async def create_thread() -> Thread:
    client = ai.beta.threads
    return await client.create()


class APIProxy(APIRouter):
    """
    A class representing an API proxy that provides access to various OpenAI API clients and tools.
    """

    @property
    def ai(self):
        """
        OpenAI API client.
        """
        return self.stack.ai

    @property
    def retriever(self):
        """
        Pinecone API client.
        """
        return self.stack.db

    @property
    def stack(self):
        """
        Stack of tools and messages.
        """
        return Stack()

    @handle  # Thread
    async def create_thread(self, stack: Stack = Depends(Stack)) -> Thread:
        client = stack.ai.beta.threads
        if stack.messages:
            return await client.create(
                messages=[
                    ThreadMessageCreate(
                        role="user", content=m.content, file_ids=m.file_ids
                    )
                    for m in stack.messages
                ]
            )
        return await client.create()

    @handle  # Assistant
    async def create_assistant(
        self,
        *,
        model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"],
        name: str,
        instructions: str,
        file_ids: Optional[str] = None,
    ) -> Assistant:
        client = self.ai.beta.assistants
        tools: list[Tool] = [
            ToolAssistantToolsFunction(
                function=o.definition(),
                type="function",
            )
            for o in OpenAIFunction.__subclasses__()
        ] + [
            ToolAssistantToolsRetrieval(type="retrieval"),
            ToolAssistantToolsCode(type="code_interpreter"),
        ]
        if not file_ids:
            response = await client.create(
                model=model,
                name=name,
                instructions=instructions,
                tools=tools,
            )
        else:
            response = await client.create(
                model=model,
                name=name,
                instructions=await use_instruction(
                    ai=ai,
                    text=f"Provide a more detail breakdown of these instructions:\n\n{instructions}.",
                ),
                tools=tools,
                file_ids=file_ids.split(","),
            )
        image = await use_image(
            ai=ai,
            text=await use_instruction(
                ai=ai,
                text=f"Describe a creative visual representation for a chatbot named {name}. with the following instructions:\n\n{instructions}.",
            ),
        )
        kwargs = {"picture": image, **response.dict()}
        return IAssistant.parse_obj(kwargs)

    @handle  # Run
    async def create_run(self, *, assistant_id: str, thread_id: str) -> Run:
        client = self.ai.beta.threads.runs
        return await client.create(assistant_id=assistant_id, thread_id=thread_id)

    @handle  # FileObject
    async def push_file(self, *, file: UploadFile = File(...)) -> FileObject:
        client = self.ai.files
        b = await file.read()
        return await client.create(file=b, purpose="assistants")

    @handle  # ThreadMessage
    async def push_message(
        self, *, thread_id: str, content: str, file_ids: Optional[str] = None
    ) -> ThreadMessage:
        client = self.ai.beta.threads.messages
        return await client.create(
            thread_id=thread_id,
            role="user",
            content=content,
            file_ids=file_ids.split(",") if file_ids else [],
        )

    @handle
    async def stream_steps(self, *, run_id: str, thread_id: str) -> EventSourceResponse:
        client = self.ai.beta.threads.runs.steps
        stream = await client.list(run_id=run_id, thread_id=thread_id)

        async def generator():
            async for event in stream:
                if event.type == "message_creation":
                    assert isinstance(event.step_details, MessageCreationStepDetails)
                    message_id = event.step_details.message_creation.message_id
                    message = await self.ai.beta.threads.messages.retrieve(
                        thread_id=thread_id, message_id=message_id
                    )
                    for item in message.content:
                        if item.type == "text":
                            yield item.text.value
                        elif item.type == "image_file":
                            file_id = item.image_file.file_id
                            file = await self.ai.files.retrieve(file_id=file_id)
                            dct = file.dict()
                            yield dct.get("metadata", {}).get("url", file_id)
                run = await self.ai.beta.threads.runs.retrieve(
                    run_id=run_id, thread_id=thread_id
                )
                if run.status not in ("queued", "in_progress", "requires_action"):
                    yield {"event": "done", "data": "done"}
                    break
                elif event.type == "tool_calls":
                    if isinstance(event.step_details, ToolCallsStepDetails):
                        for call in event.step_details.tool_calls:
                            if isinstance(call, CodeToolCall):
                                yield call.code_interpreter.input
                                for output in call.code_interpreter.outputs:
                                    if output.type == "logs":
                                        yield output.logs
                                    elif output.type == "image":
                                        yield output.image.file_id

                    if run.required_action:
                        action = run.required_action.submit_tool_outputs
                        tool_calls = action.tool_calls
                        to_call = [t for t in tool_calls if t.type == "function"]
                        outputs: list[ToolOutput] = await asyncio.gather(
                            *[exec_function_call(call=t) for t in to_call]
                        )
                        run = await self.ai.beta.threads.runs.submit_tool_outputs(
                            run_id=run_id, thread_id=thread_id, tool_outputs=outputs
                        )
                        continue

        return EventSourceResponse(generator(), sep="\r\n")
