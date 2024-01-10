# chat/consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
from langchain.callbacks.base import BaseCallbackHandler
from asgiref.sync import AsyncToSync
from channels.layers import get_channel_layer
import json


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Send the new token to all connected WebSocket clients
        AsyncToSync(get_channel_layer().group_send)(
            "chat",
            {
                "type": "chat_message",
                "text": token,
            },
        )

    def on_llm_end(self,
                   response,
                   *,
                   run_id,
                   parent_run_id,
                   **kwargs) -> None:
        # This function is called when the stream ends
        self.send_end_message()

    def send_end_message(self):
        # Send the "end" message
        AsyncToSync(get_channel_layer().group_send)(
            "chat",
            {
                "type": "chat_message",
                "text": json.dumps({"stream_end": True}),
            },
        )


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add(
            "chat",
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            "chat",
            self.channel_name
        )

    async def chat_message(self, event):
        text = event["text"]
        await self.send(text_data=json.dumps({
            "message": text
        }))
