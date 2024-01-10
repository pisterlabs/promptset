from typing import Dict, Union, Any, List

from langchain.callbacks.base import BaseCallbackHandler


class LoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        pass
        # self.logger = logger

    def on_chain_start(self, serialized: Dict[str, Any], **kwargs: Any) -> Any:
        pass
        # self.logger.log(f"Entering chain {serialized['name']}...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        pass
        # self.logger.log(f"Finished chain.")

    def on_function_start(self, serialized: Dict[str, Any]) -> Any:
        pass
        # self.logger.log(f"Entering {serialized['name']}...")

    def on_function_end(self) -> Any:
        pass
        # self.logger.log(f"Finished function.")

    def on_model_response(self, response: Dict[str, Any]) -> Any:
        pass
        # self.logger.log(f"Model response: {response}")

    def on_skanbot_response(self, response: str) -> Any:
        pass
        # self.logger.log(f"Skanbot response: {response}")
