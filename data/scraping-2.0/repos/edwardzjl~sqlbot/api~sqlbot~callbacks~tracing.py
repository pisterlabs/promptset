from typing import Any, Dict, Optional
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import ChatGenerationChunk, GenerationChunk, LLMResult
from loguru import logger


class TracingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for logging LLM input and output."""

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        logger.debug(
            f"on_llm_start run_id={run_id} parent_run_id={parent_run_id} prompts={prompts}"
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[GenerationChunk | ChatGenerationChunk] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        logger.trace(
            f"on_llm_new_token run_id={run_id} parent_run_id={parent_run_id} token={token} chunk={chunk}"
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        logger.debug(
            f"on_llm_end run_id={run_id} parent_run_id={parent_run_id} response={response}"
        )

    async def on_llm_error(
        self,
        error: Exception | KeyboardInterrupt,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        logger.error(
            f"on_llm_error run_id={run_id} parent_run_id={parent_run_id} error={error}"
        )
