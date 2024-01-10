from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

from datetime import datetime


class ChatBotOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip(" ").strip('"')
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)

        # If neither of the above conditions is met, return the message
        # "I do not know the answer to that question"
        return AgentFinish(
            return_values={"output": "I do not know the answer to your question. Please ask my Overloard Alex the King."},
            log=llm_output
        )

