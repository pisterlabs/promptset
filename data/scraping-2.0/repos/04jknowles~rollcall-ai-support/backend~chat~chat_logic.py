# chat/consumers.py
from langchain.callbacks.base import BaseCallbackHandler
from asgiref.sync import AsyncToSync
from channels.layers import get_channel_layer


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print("New token:", token)
        AsyncToSync(get_channel_layer().group_send)(
            "chat",
            {
                "type": "chat.message",
                "text": token,
            },
        )
