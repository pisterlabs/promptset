from typing import Any, Optional, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from queue import Queue

class QueueCallbackHandler(BaseCallbackHandler):

    def __init__(self, queue: Queue):
        self.queue = queue
    
    def on_llm_new_token(self, token: str, *, chunk: GenerationChunk | ChatGenerationChunk | None = None, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.queue.put({
            'event': 'message',
            'data': token,
            'retry': 10,
        })
    
    def on_llm_end(self, *args, **kwargs) -> Any:
        return self.queue.empty()