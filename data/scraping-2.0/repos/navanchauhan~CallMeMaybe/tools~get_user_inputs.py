from typing import List
from langchain.agents import tool

from dotenv import load_dotenv

from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType

load_dotenv()

import os

INPUTS = {}


@tool("get_desired_inputs")
def get_desired_inputs(input: str) -> dict:
    """Use this between tools to get the desired inputs for the next tool.
    You will be given a task that will be performed by an autonomous agent on behalf of a user. You will gather any necessary data from the user to complete the specified task before executing the task.
    """

    prompt = input

    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        tools=load_tools(["human"]),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )

    return agent.run(prompt)


def get_user_inputs():
    # iterate through INPUTS and populate values
    print("Done")
