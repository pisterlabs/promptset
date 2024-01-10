from typing import Any, Dict, List

from fastapi import WebSocketDisconnect
from langchain.callbacks.base import AsyncCallbackHandler
from loguru import logger
from starlette.websockets import WebSocketDisconnect
from websockets import ConnectionClosed

from src.app.utils import wss_close_ignore_exception
from src.domain.models import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        start_resp = ChatResponse(sender="Assistant", message="", type="start")
        await self.websocket.send_json(start_resp.dict())

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="Assistant", message=token, type="stream")
        try:
            await self.websocket.send_json(resp.dict())
        except (WebSocketDisconnect, ConnectionClosed):
            pass
        except Exception as e:
            await wss_close_ignore_exception(self.websocket)

    async def on_llm_end(self, response, **kwargs: Any) -> None:
        end_resp = ChatResponse(sender="Assistant", message="", type="end")
        await self.websocket.send_json(end_resp.dict())
