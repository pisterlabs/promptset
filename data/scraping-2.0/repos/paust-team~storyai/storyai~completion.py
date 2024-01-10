import openai
from openai.types.chat import ChatCompletionMessageParam


class Completer:
    def __init__(
            self,
            openai_api_key: str,
            model_name: str,
    ):
        self._openai = openai.OpenAI(api_key=openai_api_key)
        self._model_name = model_name

    def chat(
            self,
            messages: list[ChatCompletionMessageParam],
    ) -> str:
        if self._model_name.lower() == 'mock':
            return 'mock result of chat'

        chat_completion = self._openai.chat.completions.create(messages=messages, model=self._model_name)
        content = chat_completion.choices[0].message.content
        assert content is not None
        return content
