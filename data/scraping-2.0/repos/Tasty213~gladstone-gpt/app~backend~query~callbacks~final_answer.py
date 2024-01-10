import time
from uuid import UUID, uuid4
from langchain.callbacks.base import AsyncCallbackHandler
from fastapi import WebSocket
from typing import Any, Dict, List
from messageData import MessageData

from schema.message import Message


class FinalAnswerCallback(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(
        self,
        websocket: WebSocket,
        previous_message: Message,
        message_data_table: MessageData,
    ):
        self.websocket = websocket
        self.previous_message = previous_message
        self.messageDataTable = message_data_table

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        response_message_id = str(uuid4())
        response_message_time = int(time.time() * 1000)
        start_resp = {
            "sender": "bot",
            "messageId": response_message_id,
            "previousMessageId": self.previous_message.messageId,
            "time": response_message_time,
            "type": "start",
        }
        await self.websocket.send_json(start_resp)
        self.messageDataTable.add_message(self.previous_message)

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any
    ) -> None:
        response_message_id = str(uuid4())
        response_message_time = int(time.time() * 1000)
        output_message = Message.from_langchain_result(
            outputs.get("answer"),
            outputs.get("source_documents"),
            self.previous_message.messageId,
            response_message_id,
            response_message_time,
        )

        await self.websocket.send_json(
            {
                "sender": "bot",
                "id": output_message.messageId,
                "previousMessageId": output_message.previousMessageId,
                "message": output_message.message.content,
                "sources": output_message.sources,
                "type": "end",
            }
        )

        self.messageDataTable.add_message(output_message)
