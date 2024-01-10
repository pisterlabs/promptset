from queue import SimpleQueue, Empty
from typing import Optional, Any, Dict, List, Union, Sequence
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, Document

job_done = object()


class ResponseCallback(BaseCallbackHandler):
    queue: SimpleQueue
    documents: Sequence[Document]

    def __init__(self, queue: SimpleQueue) -> None:
        self.queue = queue

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID,
                     parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        while not self.queue.empty():
            try:
                self.queue.get(block=False)
            except Empty:
                continue

    def on_llm_new_token(self, token: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        self.queue.put(token)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                   **kwargs: Any) -> Any:
        self.queue.put(job_done)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: UUID,
                     parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        self.queue.put(job_done)

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                         **kwargs: Any) -> Any:
        self.documents = documents
