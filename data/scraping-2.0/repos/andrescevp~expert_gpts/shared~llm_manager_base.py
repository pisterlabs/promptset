from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, Tool
from langchain.agents.agent_types import AgentType
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.chat_models.base import BaseChatModel
from langchain.memory.chat_memory import BaseChatMemory

from shared.llms.openai import GPT_3_5_TURBO
from shared.patterns import Singleton

logger = logging.getLogger(__name__)


@dataclass
class Cost:
    prompt: float
    completion: float


class MyTracer(BaseTracer):
    log = {"runs": []}

    def _persist_run(self, run: Run) -> None:
        self.log["runs"].append(run.dict())


class BaseLLMManager(metaclass=Singleton):
    def __init__(self, costs: Dict[str, Cost]):
        self.costs = costs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.callbacks_handler = MyTracer()

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0

    def create_chat_completion(
        self,
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Create a chat completion and update the cost.
        Args:
        messages (list): The list of messages to send to the API.
        model (str): The model to use for the API call.
        temperature (float): The temperature to use for the API call.
        max_tokens (int): The maximum number of tokens for the API call.
        Returns:
        str: The AI's response.
        """
        pass

    def create_chat_completion_with_agent(
        self,
        user_input: str,  # type: ignore
        agent_type: AgentType = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        model: str | None = None,
        agent_key: str = "default",
        temperature: float = 0,
        max_tokens: int | None = None,
        memory: Optional[BaseChatMemory] = None,
        tools: Optional[List[Tool]] = None,
    ) -> str:
        pass

    def get_llm(
        self, max_tokens, model, temperature, as_predictor: bool = False
    ) -> BaseChatModel:
        pass

    def get_agent_executor(
        self,
        llm,
        llm_key: AgentType = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory: Optional[BaseChatMemory] = None,
        tools: Optional[List[Tool]] = None,
    ) -> AgentExecutor:
        pass

    def update_cost(self, cb):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += cb.total_tokens
        self.total_prompt_tokens += cb.prompt_tokens
        self.total_completion_tokens += cb.prompt_tokens
        self.total_cost += cb.total_cost
        logger.debug(f"Total running cost: ${self.total_cost:.3f}")

    def execute_plan(
        self,
        user_input: str,  # type: ignore
        model: str | None = GPT_3_5_TURBO,
        agent_key: str = "default_plan",
        temperature: float = 0,
        max_tokens: int | None = None,
        tools: Optional[List[Tool]] = None,
    ) -> str:
        pass
