import logging
from typing import Any, Optional
from uuid import UUID

from typing import Any, Dict, List
from langchain_core.exceptions import TracerException
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tracers.stdout import FunctionCallbackHandler
from langchain_core.utils.input import get_bolded_text, get_colored_text

from langchain_core.outputs import LLMResult

class LlmDebugHandler(BaseCallbackHandler):
    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        print("LLM Start")
        logging.debug(f"LLM Start: {serialized} {prompts}")
        for i, prompt in enumerate(prompts):
            logging.debug(f"  Prompt {i}: {prompt}")
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        logging.debug(f"LLM Token: {token}")
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        logging.debug(f"LLM Result: {response}")
        for f in response.generations:
            for gen in f:
                logging.debug(f"  Generation: {gen.text}")

    def __copy__(self) -> "LlmDebugHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "LlmDebugHandler":
        """Return a deep copy of the callback handler."""
        return self
