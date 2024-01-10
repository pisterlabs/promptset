from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain.schema import AgentFinish

DEFAULT_ANSWER_PREFIX_TOKENS = ["\nFinal", " Answer", ":"]
class FinalOutputAsyncHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue, event: asyncio.Event):
        self.queue = queue
        self.event = event
        self.last_tokens = [""] * len(DEFAULT_ANSWER_PREFIX_TOKENS)
        self.answer_reached = False
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, **kwargs: Any) -> Any:
        self.answer_reached = False
        self.event.clear()
    def on_llm_new_token(self, token: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(DEFAULT_ANSWER_PREFIX_TOKENS):
            self.last_tokens.pop(0)
        if self.last_tokens == DEFAULT_ANSWER_PREFIX_TOKENS:
            self.answer_reached = True
            return
        if self.answer_reached:
            self.queue.put_nowait(token)
        
    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.event.set()
