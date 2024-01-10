from kani import ChatMessage
from kani.engines.openai import OpenAIEngine
from kani.engines.openai.models import OpenAIChatMessage
from .img_tokens import tokens_from_image_size
from .models import OpenAIVisionChatMessage
from ...parts import ImagePart


class OpenAIVisionEngine(OpenAIEngine):
    """Engine for using vision models on the OpenAI API.

    This engine supports all vision-language models, chat-based models, and fine-tunes. It is a superset of the base
    :class:`~kani.engines.openai.OpenAIEngine`.
    """

    def __init__(self, api_key: str = None, model="gpt-4-vision-preview", *args, **kwargs):
        super().__init__(api_key, model, *args, **kwargs)
        # GPT-4 visual alpha always includes a 54-token system prompt
        if model.endswith("visual"):
            self.token_reserve = 54

    def message_len(self, message: ChatMessage) -> int:
        mlen = 7
        for part in message.parts:
            if isinstance(part, ImagePart):
                mlen += tokens_from_image_size(part.size)
            else:
                mlen += len(self.tokenizer.encode(str(part)))
        if message.name:
            mlen += len(self.tokenizer.encode(message.name))
        if message.function_call:
            mlen += len(self.tokenizer.encode(message.function_call.name))
            mlen += len(self.tokenizer.encode(message.function_call.arguments))
        return mlen

    @staticmethod
    def translate_messages(messages: list[ChatMessage], cls: type[OpenAIChatMessage] = OpenAIVisionChatMessage):
        return OpenAIEngine.translate_messages(messages, cls)
