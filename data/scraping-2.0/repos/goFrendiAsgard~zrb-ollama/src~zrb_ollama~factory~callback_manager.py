from zrb.helper.typing import Any
from zrb.helper.accessories.color import colored
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
from .schema import CallbackManagerFactory
from ..task.any_prompt_task import AnyPromptTask

import sys


class ZrbStderrCallbackHandler(StreamingStdOutCallbackHandler):

    def __init__(self) -> None:
        super().__init__()
        self._is_first_token = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        shown_text = '\n    '.join(token.split('\n'))
        if self._is_first_token:
            shown_text = ''.join(['    ', shown_text])
        print(
            colored(shown_text, attrs=['dark']),
            file=sys.stderr, end='', flush=True
        )
        self._is_first_token = False


def callback_manager_factory() -> CallbackManagerFactory:
    def create_callback_manager_factory(
        task: AnyPromptTask
    ) -> CallbackManager:
        return CallbackManager([ZrbStderrCallbackHandler()])
    return create_callback_manager_factory
