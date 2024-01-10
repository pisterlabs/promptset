from datetime import datetime

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

from constants import MODEL

load_dotenv()

from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI as OpenAIChat
from langchain.prompts import MessagesPlaceholder
from langchain.agents import (
    AgentType,
    AgentExecutor,
    OpenAIFunctionsAgent,
)
import os
from dotenv import load_dotenv

load_dotenv()
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


class PostRatingLLM:
    RATING_TEMPLATE = """Evaluate the following Reddit post based on the following criteria:

    1. Does the post provide valuable information or resources that could help someone become an expert in AI?
    2. Does the post contain the latest developments or updates in the field of AI and Language Learning Models (LLMs)?
    3. Would the post be interesting and useful to anyone aspiring to become an expert in AI, regardless of whether they are a developer or not?

    Please rate the post on a scale of 1-10 for each criterion, with 1 being 'not at all' and 10 being 'extremely'

    Post Title: {post_title}
    Post Body: {post_body}
    Post Comments: {post_comments}

    Your final output should only be a single number rating.
    """

    def __init__(self):
        self._set_llm()

    def _set_llm(self):
        MEMORY_KEY = "chat_history"
        model_verbose = False
        if os.environ.get("DEBUG") and os.environ.get("DEBUG").lower() == "true":
            model_verbose = True
        system_message = SystemMessage(
            content=f"You are a helpful AI Rating assistant. Given a string you extract the final rating out of the string. While you have not been trained on data past 2021, you can search for that data online using tools. The current date is {datetime.now()}"
        )
        llm = OpenAIChat(model=MODEL, temperature=0)
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)],
        )
        agent = OpenAIFunctionsAgent(
            llm=llm,
            tools=self._get_tools(),
            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            prompt=prompt,
            verbose=model_verbose,
            max_iterations=6,
        )
        llm_memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY, return_messages=True
        )
        self.agent = AgentExecutor(
            agent=agent, tools=self._get_tools(), memory=llm_memory, verbose=True
        )

    @staticmethod
    def _get_tools():
        search = GoogleSearchAPIWrapper()
        return [
            Tool(
                name="Search",
                func=search.run,
                description="useful for finding the latest information after 2021",
            ),
            WriteFileTool(),
            ReadFileTool(),
        ]

    def rate(self, post_title, post_body, post_comments):
        rating_string = self.agent.run(
            self.RATING_TEMPLATE.format(
                post_title=post_title,
                post_body=post_body,
                post_comments=post_comments,
            )
        )

        short_rating = self.agent.run(f"What is the final rating in the following message. The answer should be a float or integer:\n\n{rating_string}")

        for _word in short_rating.split():
            if _word.endswith("."):
                _word = _word[:-1]
            if isfloat(_word):
                return float(_word)

        return -1
