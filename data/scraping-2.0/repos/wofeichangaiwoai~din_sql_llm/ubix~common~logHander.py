from typing import Any

from langchain.callbacks.base import BaseCallbackHandler


class LogHandler(BaseCallbackHandler):
    def __init__(self, name):
        self.name = name or "LogHandler"

    def on_text(
            self,
            text: str,
            **kwargs: Any,
    ) -> Any:
        print(f"======== Here is the {self.name} info Begin=========")
        print(text)
        print(f"======== Here is the {self.name} info End=========")

