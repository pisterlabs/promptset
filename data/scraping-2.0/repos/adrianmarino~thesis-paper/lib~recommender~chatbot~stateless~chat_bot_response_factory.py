from .chat_bot_response import ChatBotResponse
import logging
from langchain_core.messages import SystemMessage


class ChatBotResponseFactory:
    def __init__(self, output_parser, template_factory):
        self._output_parser    = output_parser
        self._template_factory = template_factory
        self._logger           = logging.getLogger(self.__class__.__name__)

    def create(self, params, response):
        metadata = self._output_parser.parse(response)
        prompt = self._template_factory.invoke(params)

        metadata['params'] = params
        metadata['prompts'] = list(map(lambda it: {'type': it.__class__.__name__, 'content':  it.content}, prompt.messages))

        return ChatBotResponse(response, metadata)
