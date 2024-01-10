from typing import Any, Dict, List
import string
from langchain.callbacks.base import AsyncCallbackHandler
from schema import ChatResponse

class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.token_buffer = ""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Add the new token to the buffer
        self.token_buffer += token
        words = self.token_buffer.split()
        # if buffer contain 5-7 words, send it to the client
        if len(words) >= 5 and len(words) <= 7:
            if words[-1][-1] not in string.punctuation:
                message_to_send = " ".join(words[:-1])
                self.token_buffer = words[-1]
            else:
                message_to_send = " ".join(words)
                self.token_buffer = ""

            resp = ChatResponse(sender="interviewer", message=" "+message_to_send+" ", type="stream")
            await self.websocket.send_json(resp.dict())
    
    async def flush_buffer(self):

        # Just send the remaining words
        words = self.token_buffer.split()
        message_to_send = " ".join(words)
        resp = ChatResponse(sender="interviewer", message=" "+message_to_send+" ", type="stream")
        self.token_buffer = ""
        await self.websocket.send_json(resp.dict())
    

    async def send_specific_information(self, info):
        data = f"""/* QUESTION:\n{info} */"""
        resp = ChatResponse(sender="interviewer", message=data, type="info")
        await self.websocket.send_json(resp.dict())