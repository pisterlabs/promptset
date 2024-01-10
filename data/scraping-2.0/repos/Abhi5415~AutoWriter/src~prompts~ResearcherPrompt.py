from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever

import time
from typing import Any, Callable, List, Union

from pydantic import BaseModel

from prompts.PromptGenerator import get_prompt
from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever

from utils.StageReturnType import StageReturnType
from BaseContent import BaseContent


class ResearcherPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196

    def construct_full_prompt(self, content: BaseContent, feedback: Union[str, None]) -> str:
        prompt_start = (
            "Your decisions must always be made independently "
            "Play to your strengths as an LLM and pursue simple "
            "strategies with no legal complications.\n"
            "If you have completed all your tasks, make sure to "
            'use the "finish" command.'
        )
        # Construct full prompt
        full_prompt = (
            f"You are a technology expert who is conducting research for a blog post for the internet. \n{prompt_start}\n"
        )

        # goals
        full_prompt += f"The goals of the article are as follows: {content.goals}\n"
        full_prompt += f"The audience of the article are {content.audience}\n"
        full_prompt += f"The article's topic should be {content.title}\n"
        full_prompt += f"The article's tone should be {content.tone}\n"
        full_prompt += f"The article's length should be about 600 words\n"


        # todo research questions
        if len(content.todo_questions) > 0:
          full_prompt += f"You have yet to conduct research on the following questions:\n"
          for i, question in enumerate(content.todo_questions):
              full_prompt += f"{i}. {question}\n"
          full_prompt += "\n\n"

        # research results
        if len(content.research) > 0:
            full_prompt += "You have already conducted some research and have found the following information:\n"
            for question, answer in content.research.items():
                full_prompt += f"Question: {question}: {answer}\n"
            full_prompt += "\n\n"

        # feedback
        if feedback:
            full_prompt += f"You've received the following feedback on your work so far: {feedback}\n\n"

        # directives
        full_prompt += """
You should ask questions and add them to your research list. 
You should ask the human for feedback if you're unclear on what topics to research or need some guidance.
You're finished when you've answered all the questions on your research list and have atleast 5 question and answer results.
You should also receive positive feedback from the human before you finish.
        """

        full_prompt += f"\n\n{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["content"], kwargs["feedback"]))
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}"
        )
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
