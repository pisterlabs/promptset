import logging
from typing import Any, Dict, List
from uuid import UUID
import utilities.token_helper as token_helper
from langchain.callbacks.base import BaseCallbackHandler

class DebugCallbackHandler(BaseCallbackHandler):
    def on_text(self, text: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        just_prompt = text.lstrip("Prompt after formatting:\n\x1b[32;1m\x1b[1;3m")
        just_prompt = just_prompt.rstrip("\x1b[0m")

        logging.debug(f"on_text counted {token_helper.num_tokens_from_string(just_prompt)} tokens")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, **kwargs: Any) -> Any:  
        if 'context' in inputs:
            total_message = f"{inputs['question']}\n\n{inputs['context']}"
        else:
            total_message = inputs['question']

        logging.debug(f"on_chain_start counted {token_helper.num_tokens_from_string(total_message)} tokens")