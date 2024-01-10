from dataclasses import dataclass
from typing import Optional

from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.schema import BaseMemory
from langchain.tools.plugin import AIPlugin, AIPluginTool
from qgis.core import QgsTask

from askgis.lib.chain import GISTool
from askgis.lib.context import Context


@dataclass
class ChatResult:
    answer: str


class ChatTask(QgsTask):
    def __init__(
        self,
        description: str,
        question: str,
        context: Context,
        api_key: str,
        memory: BaseMemory,
    ) -> None:
        super().__init__(description)
        self._question = question
        self._context = context
        self._api_key = api_key
        self._memory = memory
        self._exception: Optional[Exception] = None
        self._result: Optional[ChatResult] = None

    @property
    def result(self) -> Optional[ChatResult]:
        return self._result

    def run(self) -> bool:
        try:
            llm = OpenAI(temperature=0.7, openai_api_key=self._api_key)

            tools = load_tools(["llm-math"], llm=llm)
            agent = initialize_agent(
                [
                    GISTool(context=self._context, llm=llm),
                    *NLAToolkit.from_llm_and_ai_plugin_url(
                        llm, "https://www.klarna.com/.well-known/ai-plugin.json"
                    ).get_tools(),
                    *tools,
                ],
                llm,
                AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self._memory,
                verbose=True,
            )
            answer = agent.run(self._question)
            self._result = ChatResult(answer=answer)
            return True
        except Exception as e:
            self._exception = e
            return False

    def finished(self, result: bool) -> None:
        if not result and self._exception:
            raise self._exception
