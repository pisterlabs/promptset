from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Optional
class CallbackManager(BaseCallbackHandler):
    """Class to manage callback methods for the program."""
    
    def __init__(self):
        """Initialize CallbackManager with empty last_execution list."""
        self.last_execution = []

    def clear(self):
        """Clear the last_execution list."""
        self.last_execution = []
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.last_execution.append(token)
