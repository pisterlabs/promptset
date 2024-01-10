import logging
import asyncio 
import os
from langchain.agents import tool
from dotenv import load_dotenv

from langchain.agents.agent_toolkits import GmailToolkit

from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType

load_dotenv()
toolkit = GmailToolkit()

tools = toolkit.get_tools()

my_information = "Use this information whenever needed User information " + open("info.txt").read() + "  . Your task "


@tool("email tasks")
def email_tasks(input: str) -> bool:
    """draft/send/search/get email and return whatever you get.
    the input to this tool is the prompt to the gmail toolkit.

    Re-order the tasks in the prompt to change the order in which they are executed.

    Re organise the the input to the tool to pass all information needed to complete the task.

    should  use this tool as many times needed to complete the task.

    for example, `send an email to grsi2038@colorado.edu asking him if he is still looking for a job and that he should continue doing whatever he his doing because he will eventually find it` will email grsi2038@colorado.edu
    """
    prompt = my_information + input
    #print(input) 

    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )

    return agent.run(prompt)
    
