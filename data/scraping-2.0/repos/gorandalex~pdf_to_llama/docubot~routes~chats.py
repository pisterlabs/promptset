from typing import List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.orm import Session
from langchain.vectorstores import FAISS
import pickle
import os
from dotenv import load_dotenv
from config import BASE_DIR

from docubot.database.connect import get_db
from docubot.database.models import UserRole, User
from docubot.schemas.chats import ChatPublic, CreateChatRequest, CreateChatResult
from docubot.repository import chats as repository_chats
from docubot.repository import documents as repository_documents
from docubot.repository import users_tokens as repository_users_tokens
from docubot.services.llm import send_message_to_llm
from docubot.utils.filters import UserRoleFilter
from docubot.services.auth import get_current_active_user
from docubot.services.pdf_to_vectorstore import pdf_to_vectorstore


load_dotenv()

router = APIRouter(prefix='/documents/chats', tags=["Document chats"])


@router.post("/", response_model=ChatPublic, status_code=status.HTTP_201_CREATED)
async def create_chat(
        document_id: int,
        body: CreateChatRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:

    document = await repository_documents.get_document_by_id(document_id, db)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found document")
    
    total_user_tokens = await repository_users_tokens.get_total_user_tokens(current_user.id, db)
    if total_user_tokens > 100000:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="You've used all daily tokens. We are waiting for you tomorrow")

    path_to_vectorstore = os.path.join(BASE_DIR,'storage', f"{document.public_id}.pkl")
    if os.path.exists(path_to_vectorstore):
        with open(path_to_vectorstore,"rb") as f:
            vectorstore = pickle.load(f)
    else:
        vectorstore = await pdf_to_vectorstore(os.path.join(BASE_DIR,'storage', f"{document.public_id}.pdf"))
        # raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found document")
    
    # todo send message to llm
    answer, cb = await send_message_to_llm(vectorstore, body.question)
    
    users_tokens = await repository_users_tokens.add_user_tokens(user_id=current_user.id, user_tokens=cb.total_tokens, db=db)

    return await repository_chats.create_chat(
        current_user.id,
        document_id,
        body.question.strip(),
        answer,
        db
    )


@router.get(
    '/',
    response_model=List[ChatPublic],
    description='No more than 100 requests per minute',
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
async def get_chats_by_document_or_user_id(
        document_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 10,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:
    if document_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Document_id must be provided")

    return await repository_chats.get_chats_by_document_or_user_id(
        current_user.id,
        document_id,
        skip,
        limit,
        db
    )


@router.get("/{chat_id}", response_model=ChatPublic)
async def get_chat(
        chat_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:

    chat = await repository_chats.get_chat_by_id(chat_id, db)

    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    return chat



@router.delete('/{chat_id}', dependencies=[Depends(UserRoleFilter(UserRole.moderator))])
async def remove_chat(
        chat_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_active_user)
) -> Any:

    chat = await repository_chats.remove_chat(chat_id, db)

    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    return chat
