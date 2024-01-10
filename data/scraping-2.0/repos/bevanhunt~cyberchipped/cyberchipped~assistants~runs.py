import asyncio
from typing import Any, Callable, Optional, Union

from openai.types.beta.threads.run import Run as OpenAIRun
from pydantic import BaseModel, Field, field_validator

import cyberchipped.utilities.tools
from cyberchipped.requests import Tool
from cyberchipped.tools.assistants import AssistantTools, CancelRun
from cyberchipped.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from cyberchipped.utilities.logging import get_logger
from cyberchipped.utilities.openai import get_client

from .assistants import Assistant
from .threads import Thread

logger = get_logger("Runs")


class Run(BaseModel, ExposeSyncMethodsMixin):
    thread: Thread
    assistant: Assistant
    instructions: Optional[str] = Field(
        None, description="Replacement instructions to use for the run."
    )
    additional_instructions: Optional[str] = Field(
        None,
        description=(
            "Additional instructions to append to the assistant's instructions."
        ),
    )
    tools: Optional[list[Union[AssistantTools, Callable]]] = Field(
        None, description="Replacement tools to use for the run."
    )
    additional_tools: Optional[list[AssistantTools]] = Field(
        None,
        description="Additional tools to append to the assistant's tools. ",
    )
    run: OpenAIRun = None
    data: Any = None

    @field_validator("tools", "additional_tools", mode="before")
    def format_tools(cls, tools: Union[None, list[Union[Tool, Callable]]]):
        if tools is not None:
            return [
                (
                    tool
                    if isinstance(tool, Tool)
                    else cyberchipped.utilities.tools.tool_from_function(tool)
                )
                for tool in tools
            ]

    @expose_sync_method("refresh")
    async def refresh_async(self):
        client = get_client()
        self.run = await client.beta.threads.runs.retrieve(
            run_id=self.run.id, thread_id=self.thread.id
        )

    @expose_sync_method("cancel")
    async def cancel_async(self):
        client = get_client()
        await client.beta.threads.runs.cancel(
            run_id=self.run.id, thread_id=self.thread.id
        )

    async def _handle_step_requires_action(self):
        client = get_client()
        if self.run.status != "requires_action":
            return
        if self.run.required_action.type == "submit_tool_outputs":
            tool_outputs = []
            tools = self.get_tools()

            for tool_call in self.run.required_action.submit_tool_outputs.tool_calls:
                try:
                    output = cyberchipped.utilities.tools.call_function_tool(
                        tools=tools,
                        function_name=tool_call.function.name,
                        function_arguments_json=tool_call.function.arguments,
                    )
                except CancelRun as exc:
                    logger.debug(f"Ending run with data: {exc.data}")
                    raise
                except Exception as exc:
                    output = f"Error calling function {tool_call.function.name}: {exc}"
                    logger.error(output)
                tool_outputs.append(
                    dict(tool_call_id=tool_call.id, output=output or "")
                )

            await client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id, run_id=self.run.id, tool_outputs=tool_outputs
            )

    def get_instructions(self) -> str:
        if self.instructions is None:
            instructions = self.assistant.get_instructions() or ""
        else:
            instructions = self.instructions

        if self.additional_instructions is not None:
            instructions = "\n\n".join([instructions, self.additional_instructions])

        return instructions

    def get_tools(self) -> list[AssistantTools]:
        tools = []
        if self.tools is None:
            tools.extend(self.assistant.get_tools())
        else:
            tools.extend(self.tools)
        if self.additional_tools is not None:
            tools.extend(self.additional_tools)
        return tools

    async def run_async(self) -> "Run":
        client = get_client()

        create_kwargs = {}

        if self.instructions is not None or self.additional_instructions is not None:
            create_kwargs["instructions"] = self.get_instructions()

        if self.tools is not None or self.additional_tools is not None:
            create_kwargs["tools"] = self.get_tools()

        self.run = await client.beta.threads.runs.create(
            thread_id=self.thread.id, assistant_id=self.assistant.id, **create_kwargs
        )

        try:
            await asyncio.wait_for(self._run_loop(), timeout=60)
        except asyncio.TimeoutError:
            if self.run.status != "completed":
                # Cancel the run if it's not completed
                await client.beta.threads.runs.cancel(
                    run_id=self.run.id, thread_id=self.thread.id
                )
                self.data = "Run cancelled due to timeout."
            else:
                self.data = "Run already completed; no need to cancel."
            await self.refresh_async()

    async def _run_loop(self):
        while self.run.status in ("queued", "in_progress", "requires_action"):
            if self.run.status == "requires_action":
                await self._handle_step_requires_action()
            await asyncio.sleep(0.1)
            await self.refresh_async()
