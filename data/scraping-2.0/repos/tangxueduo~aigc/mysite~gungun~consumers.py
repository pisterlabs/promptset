# gungun/consumers.py
import os
import json
from loguru import logger
from channels.generic.websocket import WebsocketConsumer
from langchain.chat_models import ChatOpenAI


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        logger.warning(f"text_data_json: {text_data_json}") 
        message = text_data_json["message"]
        _key = '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'
        os.environ['OPENAI_API_KEY'] = _key
        chat_model = ChatOpenAI()
        resp = chat_model.predict(message)

        self.send(text_data=json.dumps({"message": resp}))