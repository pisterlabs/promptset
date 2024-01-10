"""Contains the core code for running SparksAI"""
import logging
import os
from typing import AsyncIterator

from langchain import agents
from langchain.agents import openai_assistant

from SparksAI import config
from SparksAI import tools
from SparksAI import databases
from SparksAI.swarm import Swarm
from SparksAI.memory import AIMemory
from SparksAI.async_helpers import AsyncMessageIterator

logger = logging.getLogger(__name__)


class SparksAI:
    """Core SparksAI Class, handles noticing messages and generating replies"""

    def __init__(self):
        logging.info("Initialising SparksAI")
        self.swarm = Swarm()
        self.memory = AIMemory(
            databases.FireBaseStrategy(os.getenv("FIREBASE_TABLE_ID"))
        )
        self.agent = openai_assistant.OpenAIAssistantRunnable(
            assistant_id=config.TAV_DECIDER_ID, as_agent=True
        )

    async def notice_message(
        self, username: str, msg: str, run_id: str
    ) -> AsyncIterator:
        self.memory.get_convo_mem(username=username).add_user_message(msg)

        decider = agents.AgentExecutor(
            agent=self.agent,
            tools=[
                tools.ImageAgentTool(),
                tools.ResearchAgentTool(),
            ],
            verbose=True,
        )

        input_msg = {"content": msg}

        thread_id = self.memory.reterive_user_thread_id(username=username)

        if thread_id:
            logger.info("Found existing thread id: %s for user %s", thread_id, username)
            input_msg["thread_id"] = thread_id
        else:
            logger.info("Can not find thread id for username %s", username)

        logger.info("%s: getting response : %s", run_id, input_msg)

        response = await decider.ainvoke(input_msg)

        logger.info("%s: response : %s", run_id, response)

        self.memory.update_user_details(username, response["thread_id"])

        return AsyncMessageIterator(response["output"], 20)
