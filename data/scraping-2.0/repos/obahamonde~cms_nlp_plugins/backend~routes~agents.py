import asyncio
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai.types.beta import Assistant
from openai.types.beta.assistants import AssistantFile
from openai.types.beta.assistants.file_delete_response import \
    FileDeleteResponse
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

app = APIRouter()
ai = AsyncOpenAI()


class ToolOutputList(BaseModel):
    __root__: list[ToolOutput] = Field(..., title="Tool Outputs")


@app.post("/api/assistant", response_model=Assistant)
async def create_assistant(
    name: str,
    instructions: str,
    model: Literal["gpt-3.5-turbo-1106", "gpt-4-1106-preview"] = "gpt-3.5-turbo-1106",
):
    """
    Create a new assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    return response


@app.delete("/api/assistant/{assistant_id}", response_model=None)
async def delete_assistant(assistant_id: str):
    """
    Delete an assistant.
    """
    assistants = ai.beta.assistants
    await assistants.delete(assistant_id=assistant_id)


@app.get("/api/assistant/{assistant_id}", response_model=Assistant)
async def retrieve_assistant(assistant_id: str):
    """
    Retrieve an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.retrieve(assistant_id=assistant_id)
    return response


@app.get("/api/assistant", response_class=StreamingResponse)
async def retrieve_all_assistants():
    """
    Retrieve all assistants.
    """
    assistants = ai.beta.assistants
    response = await assistants.list()

    async def generator():
        async for assistant in response:
            yield f"data: {assistant.json()}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.put("/api/assistant/files/{assistant_id}", response_model=AssistantFile)
async def attach_file(assistant_id: str, file_id: str):
    """
    Attach a file to an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.files.create(
        assistant_id=assistant_id,
        file_id=file_id,
    )
    return response


@app.delete("/api/assistant/files/{assistant_id}", response_model=FileDeleteResponse)
async def detach_file(assistant_id: str, file_id: str):
    """
    Detach a file from an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.files.delete(
        assistant_id=assistant_id,
        file_id=file_id,
    )
    return response


@app.get("/api/assistant/files/{assistant_id}", response_class=StreamingResponse)
async def retrieve_attached_files(assistant_id: str):
    """
    Retrieve all files attached to an assistant.
    """
    assistants = ai.beta.assistants
    response = await assistants.files.list(assistant_id=assistant_id)

    async def generator():
        async for file in response:
            yield f"data: {file.json()}"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/run/{thread_id}", response_model=Run)
async def run_thread(thread_id: str, assistant_id: str):
    """
    Run a thread.
    """
    threads = ai.beta.threads
    response = await threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return response


@app.get("/api/run/{thread_id}", response_model=Run)
async def retrieve_run(thread_id: str, run_id: str):
    """
    Retrieve a run.
    """
    threads = ai.beta.threads
    response = await threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id,
    )
    return response


@app.get("/api/run", response_class=StreamingResponse)
async def retrieve_all_runs(thread_id: str):
    response = await ai.beta.threads.runs.list(thread_id=thread_id)

    async def generator():
        async for run in response:
            yield f"data: {run.json()}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/run/events/{run_id}", response_class=StreamingResponse)
async def run_events(run_id: str, thread_id: str, outputs: str):
    """
    Attach a tool to a thread.
    """

    async def generator():
        runner = await retrieve_run(thread_id=thread_id, run_id=run_id)
        threads = ai.beta.threads
        while runner.status not in ("completed", "failed", "cancelled", "expired"):
            if runner.status == "requires_action":
                run = await threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=ToolOutputList.parse_raw(outputs).dict(),  # type: ignore
                )
                yield f"data: {run.json()}\n\n"
                await asyncio.sleep(0.5)
                continue
            if runner.status == "cancelling":
                response = await threads.runs.steps.list(
                    thread_id=thread_id, run_id=runner.id
                )
                async for step in response:
                    yield f"data: {step.json()}"
                await asyncio.sleep(0.5)
                continue
            if runner.status == "queued":
                yield f"data: {runner.json()}\n\n"
                await asyncio.sleep(1)
                continue
            if runner.status == "in_progress":
                response = await threads.runs.steps.list(
                    thread_id=thread_id, run_id=runner.id
                )
                async for step in response:
                    yield f"data: {step.json()}\n\n"
                await asyncio.sleep(0.5)
                continue
            if runner.status == "completed":
                response = await threads.runs.steps.list(
                    thread_id=thread_id, run_id=runner.id
                )
                async for step in response:
                    yield f"data: {step.json()}\n\n"
                await asyncio.sleep(0.5)
                continue
            if runner.status in ("failed", "cancelled", "expired"):
                yield f"data: {runner.json()}\n\n"
                break
            await asyncio.sleep(1)
            runner = await retrieve_run(thread_id=thread_id, run_id=run_id)
            yield f"data: {runner.json()}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream")
