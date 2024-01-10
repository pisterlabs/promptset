from openai import OpenAI

from fastapi import APIRouter, Depends, status

from ..user.util import AuthenticatedUserDep
from ...database import SessionDep
from ...model.chat import ChatRead, Chat, ChatCreate
from ...model.ai import AI
from ...ai import default_ai
from ...util.describe import describe
from ...config import settings

client = OpenAI(api_key = settings.openai_api_key)
router = APIRouter(prefix="/chat")

@describe(
""" Creates a new chat in the database.

Args:
    new_chat (ChatCreate): The chat to create.
    ai (AI): The AI to use for this chat.
    current_user (AuthenticatedUserDep): The current user.
    session (SessionDep): The database session.

Returns:
    The created chat.
""")
@router.post("", status_code=status.HTTP_201_CREATED)
def create_chat(*, new_chat: ChatCreate, ai:AI = Depends(default_ai), current_user: AuthenticatedUserDep, session: SessionDep) -> ChatRead:
    chat = Chat(**new_chat.model_dump(), user_id=current_user.id, assistant_id=ai.id)
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat