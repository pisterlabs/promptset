from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from schema.openai_completion_schema import (
    UpdateCompletionRequest,
    UpdateCompletionResponse,
)
from schema.openai_chat_completion_schema import (
    GetChatCompletionHistoryResponse,
    UpdateChatCompletionRequest,
    UpdateChatCompletionResponse,
)
from service import openai_completion_service
from service import openai_chat_completion_service
from sqlalchemy.orm import Session
from typing import Optional
from util.db_util import get_db


router = APIRouter(
    prefix="/llm/openai",
    tags=["llm openai"],
    responses={404: {"description": "Not found"}},
)


@router.delete("/chat-completion/{chat_completion_id}")
def delete_chat_completion(
    chat_completion_id: int, db: Session = Depends(get_db)
) -> None:
    return openai_chat_completion_service.delete_chat_completion(chat_completion_id, db)


@router.delete("/chat-completion")
def delete_user_chat_completions(username: str, db: Session = Depends(get_db)) -> None:
    return openai_chat_completion_service.delete_user_chat_completions(username, db)


@router.get("/chat-completion", response_model=GetChatCompletionHistoryResponse)
def get_user_template_chat_completion_history(
    username: str, template_id: Optional[int] = -1, db: Session = Depends(get_db)
) -> GetChatCompletionHistoryResponse:
    return openai_chat_completion_service.get_user_template_chat_completion_history(
        username, template_id, db
    )


@router.get("/chat-completion-stream", response_class=StreamingResponse)
def stream_chat_completion(
    chat_completion_id: int, test_mode: bool, db: Session = Depends(get_db)
) -> StreamingResponse:
    (
        chat_completion,
        chat_model,
        messages,
    ) = openai_chat_completion_service.prepare_chat_completion(chat_completion_id, db)
    if test_mode:
        return StreamingResponse(
            openai_completion_service.generate_test_stream(str(messages)),
            media_type="text/event-stream",
        )
    else:
        return StreamingResponse(
            openai_completion_service.generate_stream_for_chat_model(
                chat_completion, chat_model, messages, db
            ),
            media_type="text/event-stream",
        )


@router.get("/completion-stream", response_class=StreamingResponse)
def stream_completion(
    completion_id: int, test_mode: bool, db: Session = Depends(get_db)
) -> StreamingResponse:
    llm, prompt = openai_completion_service.prepare_completion(completion_id, db)
    if test_mode:
        return StreamingResponse(
            openai_completion_service.generate_test_stream(prompt),
            media_type="text/event-stream",
        )
    else:
        return StreamingResponse(
            openai_completion_service.generate_stream(llm, prompt),
            media_type="text/event-stream",
        )


@router.put("/chat-completion", response_model=UpdateChatCompletionResponse)
def update_chat_completion(
    request: UpdateChatCompletionRequest,
    db: Session = Depends(get_db),
) -> UpdateChatCompletionResponse:
    return openai_chat_completion_service.create_update_chat_completion(request, db)


@router.put("/completion", response_model=UpdateCompletionResponse)
def update_completion(
    request: UpdateCompletionRequest,
    db: Session = Depends(get_db),
) -> UpdateCompletionResponse:
    return openai_completion_service.create_update_completion(request, db)
