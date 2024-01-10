from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage

import logging

from config.config import AgentConfig
from utils import agent_tools

logger = logging.getLogger(__name__)


class AgentExecutorService:
    _agent_executor: AgentExecutor = None

    @classmethod
    def initialize(cls):

        tools = agent_tools.get_enabled_tools()
        llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")

        cls._agent_executor = create_conversational_retrieval_agent(
            llm, tools, verbose=True, system_message=SystemMessage(content=AgentConfig.SYSTEM_MESSAGE),
            max_token_limit=20000
        )

    @classmethod
    def execute(cls, query: str) -> str:
        if cls._agent_executor is None:
            AgentExecutorService.initialize()

        query = f"In the context of Neuradev, {query}"
        logger.info(f'Query: {query}')

        result = cls._agent_executor({"input": query})

        logger.info(f'Agent Response: {result["output"]}')

        return result["output"]
