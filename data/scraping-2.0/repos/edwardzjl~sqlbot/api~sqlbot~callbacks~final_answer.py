from typing import Any, Optional
from uuid import UUID

from fastapi import WebSocket
from langchain.schema import AgentFinish
from langchain.schema.output import LLMResult

from sqlbot.callbacks.base import WebsocketCallbackHandler
from sqlbot.schemas import ChatMessage

DEFAULT_ANSWER_PREFIX_TOKENS = ["Final", "Answer", ":"]


class StreamingFinalAnswerCallbackHandler(WebsocketCallbackHandler):
    """Callback handler for streaming final anwer in agents.
    A modified version of `langchain.callbacks.streaming_stdout_final_only.FinalStreamingStdOutCallbackHandler`
    from which I removed some features which I don't think is correctly designed.
    """

    def __init__(
        self,
        websocket: WebSocket,
        conversation_id: str,
        answer_prefix_tokens: Optional[list[str]] = None,
        strip_tokens: bool = True,
    ) -> None:
        """Instantiate FinalStreamingStdOutCallbackHandler.

        Args:
            answer_prefix_tokens: Token sequence that prefixes the answer.
                Default is ["Final", "Answer", ":"]
            strip_tokens: Ignore white spaces and new lines when comparing
                answer_prefix_tokens to last tokens? (to determine if answer has been
                reached)
        """
        super().__init__(websocket, conversation_id)
        if answer_prefix_tokens is None:
            self.answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        else:
            self.answer_prefix_tokens = answer_prefix_tokens
        if strip_tokens:
            self.answer_prefix_tokens = [
                token.strip() for token in self.answer_prefix_tokens
            ]
        self.strip_tokens = strip_tokens

    def append_to_last_tokens(self, token: str) -> None:
        if self.strip_tokens:
            stripped_token = token.strip()
            if stripped_token:
                self.last_tokens.append(token.strip())
        else:
            self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)

    def check_if_answer_reached(self) -> bool:
        return self.last_tokens == self.answer_prefix_tokens

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running.
        Status reset must be done in this method, not in `on_llm_end` method.
        Because once `on_llm_error` is called, the `on_llm_end` method will never be called.
        """
        self.last_tokens = [""] * len(self.answer_prefix_tokens)
        self.answer_reached = False

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # ... if yes, then print tokens from now on
        if self.answer_reached:
            message = ChatMessage(
                id=run_id,
                conversation=self.conversation_id,
                from_="ai",
                content=token,
                type="stream/text",
            )
            await self.websocket.send_text(message.model_dump_json())
            return

        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.message_id = run_id
            self.answer_reached = True
            message = ChatMessage(
                id=run_id,
                conversation=self.conversation_id,
                from_="ai",
                content=None,
                type="stream/start",
            )
            await self.websocket.send_text(message.model_dump_json())
            return

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
        message = ChatMessage(
            id=run_id,
            conversation=self.conversation_id,
            from_="ai",
            content=None,
            type="stream/end",
        )
        await self.websocket.send_text(message.model_dump_json())

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""
        reason = finish.return_values.get("reason", None)
        if reason == "early_stopped":
            message = ChatMessage(
                id=run_id,
                conversation=self.conversation_id,
                from_="ai",
                content=finish.return_values.get("output", None),
                type="text",
            )
            await self.websocket.send_text(message.model_dump_json())

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        # TODO: this condition is a bit naive
        if self.answer_reached:
            if "intermediate_steps" in outputs:
                message = ChatMessage(
                    id=self.message_id,
                    conversation=self.conversation_id,
                    from_="ai",
                    intermediate_steps=outputs["intermediate_steps"],
                    type="info/intermediate-steps",
                )
                await self.websocket.send_text(message.model_dump_json())
