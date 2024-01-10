# Данный модуль содержит роуты для генерации статей при помощи GPT моделей
import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from openai.openai_object import OpenAIObject
from starlette import status

from .models import ArticleGenerate, GeneratedArticleResponse
from .writer import Writer
from ..articles.models import ArticleDocument
from ..users.models import UserDocument
from ...core.config import FASTAPI_CHATGPT_ALTERNATIVE_BASE, FASTAPI_CHATGPT_API_KEY
from ...core.security.roles import RolesEnum
from ...core.security.utilities import RoleChecker

router = APIRouter(prefix="/gpt-writer")

log = logging.getLogger("blogapp")


@router.post("/", response_model=GeneratedArticleResponse)
async def generate_article(
    article_data: ArticleGenerate,
    current_user: Annotated[
        UserDocument, Depends(RoleChecker(allowed_role=RolesEnum.ADMIN.value))
    ],
):
    """Генерирует статью с помощью gpt-модели"""

    if not (FASTAPI_CHATGPT_ALTERNATIVE_BASE and FASTAPI_CHATGPT_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Missing required environment variables",
        )

    # Подготовка промта
    prompt: str = Writer.prepare_article_generation_prompt(
        title=article_data.title,
        tags=article_data.tags,
        key_phrases=article_data.key_phrases,
    )

    # Получение текста и обработка ошибок
    openai_completion: OpenAIObject = await Writer.generate_article_content(
        prompt=prompt
    )
    try:
        article_text: str = openai_completion["choices"][0]["message"]["content"]
    except KeyError:
        log.error(openai_completion["Error"])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Не удалось получить текст статьи}",
        )
    log.debug(f"Текст статьи: {article_text}")
    completion_errors = Writer.check_article(article_text)
    if completion_errors:
        logging.warning(f"Проблемы с генерацией статьи: {completion_errors}")
        raise HTTPException(
            status_code=400, detail=f"Проблемы с генерацией статьи: {completion_errors}"
        )

    # Вставка документа
    article = ArticleDocument(
        title=article_data.title,
        content=article_text,
        tags=article_data.tags,
        created_at=datetime.now(),
        author=current_user,
    )
    await ArticleDocument.insert_one(article)

    return {
        "article": article,
        "tokens_used": openai_completion["usage"]["completion_tokens"],
    }
