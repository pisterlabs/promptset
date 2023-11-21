from typing import Callable, List

from promptulate.agents import BaseAgent
from promptulate.agents.web_agent.prompt import SYSTEM_PROMPT_TEMPLATE
from promptulate.hook import Hook, HookTable
from promptulate.llms import BaseLLM, ChatOpenAI
from promptulate.tools import DuckDuckGoTool
from promptulate.utils.logger import get_logger

logger = get_logger()


def _build_system_prompt(prompt) -> str:
    """Build the system prompt."""
    return SYSTEM_PROMPT_TEMPLATE.format(prompt=prompt)


class WebAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM = None,
        hooks: List[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(hooks, *args, **kwargs)
        self.llm: BaseLLM = llm or ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0.0,
            enable_default_system_prompt=False,
        )
        self.stop_sequences: List[str] = ["Observation"]
        self.websearch = DuckDuckGoTool()
        self.conversation_prompt: str = ""

    def get_llm(self) -> BaseLLM:
        return self.llm

    def _run(self, prompt: str, *args, **kwargs) -> str:
        # ErnieBot built-in network search
        if self.llm.llm_type == "ErnieBot":
            return self.llm(prompt)

        self.conversation_prompt = _build_system_prompt(prompt)
        iterations = 0

        # Loop search until find the answer
        while True:
            answer: str = self.llm(self.conversation_prompt, stop=self.stop_sequences)
            logger.info(
                f"[pne] tool agent <{iterations}> current prompt: {self.conversation_prompt}"
            )

            if "Final Answer" in answer:
                return answer.split("Final Answer:")[-1]

            query_words: str = self._find_query_words(answer)
            Hook.call_hook(
                HookTable.ON_AGENT_ACTION,
                self,
                action="websearch",
                action_input=query_words,
            )

            query_result: str = self.websearch.run(query_words)
            Hook.call_hook(
                HookTable.ON_AGENT_OBSERVATION, self, observation=query_result
            )
            self.conversation_prompt += f"Observation: {query_result}\nThought: "
            iterations += 1

    def _find_query_words(self, answer: str) -> str:
        return answer.split("Query:")[-1].replace('"', "")
