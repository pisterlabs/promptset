from typing import Any, Coroutine, Dict, List, Optional
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from sqlalchemy.orm import Session
from app.database.crud.conversation_history import create_conversation_history
from app.database.schema.conversation_history import ConversationHistoryCreate

class PostgresCallbackHandler(AsyncCallbackHandler):

    def __init__(self, session: Session, conversation_id: str):
        self.session = session
        self.conversation_id = conversation_id

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        pass
    
    async def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, **kwargs: Any) -> Coroutine[Any, Any, None]:
        # print('getting here')
        # print(outputs)
        pass