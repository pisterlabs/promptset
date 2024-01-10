import json
from typing import Optional, Any, Dict, List
from uuid import UUID

from channels.generic.websocket import AsyncWebsocketConsumer

# Import necessary classes and modules
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult, BaseMessage


# Define a custom WebSocket callback handler
class AsyncStreamingCallbackHandler(AsyncCallbackHandler):

    def __init__(self, consumer: AsyncWebsocketConsumer):
        self.consumer = consumer

    # Handle the event when a new token is received from the LLM (Language Model)
    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # Send the token as a JSON message to the WebSocket consumer
        await self.consumer.send(text_data=json.dumps({'message': token, 'type': 'debug'}))

    # Handle the event when the LLM processing is completed
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # Send a newline character as a JSON message to the WebSocket consumer
        await self.consumer.send(text_data=json.dumps({'message': '\n\n', 'type': 'debug'}))

    # Handle the event when a chat model starts processing
    async def on_chat_model_start(
        self, serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        # Do nothing
        pass
