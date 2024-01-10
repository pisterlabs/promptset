from typing import Any, Optional
from uuid import UUID

from fastapi import WebSocket
from langchain_core.agents import AgentAction
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage

from pybot.schemas import ChatMessage


class AgentActionCallbackHandler(AsyncCallbackHandler):
    """Callback handler for sending tool execution information back to user."""

    def __init__(
        self,
        websocket: WebSocket,
        conversation_id: str,
        history: BaseChatMessageHistory,
    ):
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.history = history
        self.sessions = {}

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Persist 'thought' (action log) to history.
        Action log was sent to user by streaming callback, I don't need to send it again here.
        """
        self.history.add_ai_message(action.log)
        self.sessions[run_id] = action.tool

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Send 'observation' (action result) to user, and persist to history."""
        tool = self.sessions.pop(parent_run_id, "system-observation")
        message = ChatMessage(
            id=run_id,
            conversation=self.conversation_id,
            from_="system",
            content=output,
            type="text",
        )
        await self.websocket.send_text(message.model_dump_json())
        self.history.add_message(
            SystemMessage(
                content=output,
                additional_kwargs={
                    "prefix": f"<|im_start|>{tool}\n",
                    "suffix": "<|im_end|>",
                },
            )
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.sessions.pop(parent_run_id, None)
        message = ChatMessage(
            id=run_id,
            conversation=self.conversation_id,
            from_="system",
            content=str(error),
            type="text",
        )
        await self.websocket.send_text(message.model_dump_json())
        self.history.add_message(SystemMessage(content=str(error)))
