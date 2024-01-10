import json
from channels.generic.websocket import WebsocketConsumer
from langchain.llms import Ollama

from django_streaming import settings


class ChatChannel(WebsocketConsumer):

    def __init__(self):
        super(ChatChannel, self).__init__()
        self.llm = Ollama(model="llama2", base_url=settings.LLMS_BASE_URL)
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        self.close()

    def receive(self, text_data, **kwargs):
        text_data_json = json.loads(text_data)
        expression = text_data_json['prompt']
        result = self.llm.predict(expression)
        self.send(text_data=json.dumps({
            'result': result
        }))
