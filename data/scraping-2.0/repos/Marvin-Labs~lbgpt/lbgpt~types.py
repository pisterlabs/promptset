import sys

from openai.types.chat import ChatCompletion

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class ChatCompletionAddition(ChatCompletion):
    is_exact: bool = True

    @classmethod
    def from_chat_completion(
        cls, chat_completion: ChatCompletion, is_exact: bool = True
    ) -> Self:
        return cls(**chat_completion.model_dump(), is_exact=is_exact)
