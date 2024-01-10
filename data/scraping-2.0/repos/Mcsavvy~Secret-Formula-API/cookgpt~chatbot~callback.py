"""Callbacks for the chatbot."""

from typing import Any, Dict, List, cast
from uuid import UUID

from langchain.callbacks import OpenAICallbackHandler
from langchain.schema import BaseMessage, ChatGeneration, LLMResult

from cookgpt import logging
from cookgpt.chatbot.utils import (
    convert_message_to_dict,
    num_tokens_from_messages,
)
from cookgpt.ext.config import config
from cookgpt.globals import getvar, response, setvar, user
from cookgpt.utils import utcnow


class ChatCallbackHandler(OpenAICallbackHandler):
    """tracks the cost and time of the conversation"""

    var = None
    verbose: bool = config.LANGCHAIN_VERBOSE
    _query_cost: int = 0
    raise_error = True

    def compute_completion_tokens(self, result: LLMResult, model_name: str):
        """Compute the cost of the result."""
        from cookgpt.chatbot.models import Chat

        logging.debug("Computing completion tokens...")
        ai_message = cast(ChatGeneration, result.generations[0][0]).message
        # set the id of the response
        if (response := getvar("response", Chat, None)) is not None:
            ai_message.additional_kwargs["id"] = response.pk
        ai_message_raw = convert_message_to_dict(ai_message)
        num_tokens = num_tokens_from_messages([ai_message_raw], model_name)
        # completion_cost = get_openai_token_cost_for_model(
        # model_name, num_tokens, is_completion=True
        # )
        # logging.debug("Completion cost: $%s", completion_cost)
        self.total_tokens += num_tokens
        self.completion_tokens += num_tokens
        # self.total_cost += completion_cost
        setvar("chat_cost", (self._query_cost, num_tokens))
        self._query_cost = 0

    def compute_prompt_tokens(
        self, messages: List[BaseMessage], model_name: str
    ):
        """Compute the cost of the prompt."""
        logging.debug("Computing prompt tokens...")
        messages_raw = []
        messages_raw = [convert_message_to_dict(m) for m in messages]
        # logging.debug("Messages: %s", messages_raw)
        num_tokens = num_tokens_from_messages(messages_raw, model_name)
        # prompt_cost = get_openai_token_cost_for_model(model_name, num_tokens)
        # logging.debug("Prompt cost: %s", prompt_cost)
        # self.total_tokens += num_tokens
        # self.prompt_tokens += num_tokens
        # self.total_cost += prompt_cost
        self._query_cost = num_tokens

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """create the query and response"""
        logging.info("Starting chain...")
        super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        setvar("query_time", utcnow())

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """tracks the cost of the query"""
        logging.info("Starting chat model...")
        self.compute_prompt_tokens(messages[0], "gpt-3.5-turbo-0613")

    def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """
        Run on new LLM token.
        Only available when streaming is enabled.
        """
        from cookgpt.chatbot.utils import get_stream_name
        from cookgpt.globals import current_app as app

        if self.verbose:  # pragma: no cover
            print(token, end="", flush=True)
        assert response, "No response found."
        stream = get_stream_name(user, response)
        app.redis.xadd(
            stream,
            {"token": token, "count": 1},
            maxlen=1000,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """tracks the cost of the conversation"""
        logging.info("Ending LLM...")
        setvar("response_time", utcnow())
        assert not response.llm_output, (
            "The token usage should not be in the LLM output "
            "since we are using the streaming API."
        )
        self.compute_completion_tokens(response, "gpt-3.5-turbo-0613")

    def register(self):
        """register the callback handler"""
        from langchain.callbacks.manager import openai_callback_var

        logging.debug("Registering callback handler...")
        self.var = openai_callback_var.set(self)

    def unregister(self):
        """unregister the callback handler"""
        logging.debug("Unregistering callback handler...")
        from langchain.callbacks.manager import openai_callback_var

        openai_callback_var.reset(self.var)
