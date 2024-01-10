import os
import sys
import typing
from dotenv import load_dotenv

from tools.contacts import get_all_contacts
from tools.vocode import call_phone_number
from tools.summarize import summarize
from tools.get_user_inputs import get_desired_inputs
from tools.email_tool import email_tasks
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools

from stdout_filterer import RedactPhoneNumbers

load_dotenv()

from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import BedrockChat
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.tools import WikipediaQueryRun

import argparse

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools=load_tools(["human", "wikipedia"]) + [get_all_contacts, call_phone_number, email_tasks, summarize]

tools_desc = ""
for tool in tools:
    tools_desc += tool.name + " : "  + tool.description + "\n"

def rephrase_prompt(objective: str) -> str:
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # type: ignore
    # pred = llm.predict(f"Based on these tools {tools_desc} with the {objective} should be done in the following manner (outputting a single sentence), allowing for failure: ")
    # print(pred)
    # return pred
    return f"{objective}"

with open("info.txt") as f:
    my_info = f.read()
    memory.chat_memory.add_user_message("User information to us " + my_info + " end of user information.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line argument parser example")

    parser.add_argument("--objective", type=str, help="Objective for the program")
    parser.add_argument("--verbose", type=bool, help="Verbosity of the program", default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Get the value of --objective
    objective_value = args.objective

    # Get the value of --verbose
    verbose_value = args.verbose

    # Redirect stdout to our custom class
    sys.stdout = typing.cast(typing.TextIO, RedactPhoneNumbers(sys.stdout))

    if objective_value is None:
        objective_value = input("What is your objective? ")

    OBJECTIVE = (
        objective_value
        or "Find a random person in my contacts and tell them a joke"
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # type: ignore
    #llm = BedrockChat(model_id="anthropic.claude-instant-v1", model_kwargs={"temperature":0})  # type: ignore
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Logging of LLMChains
    verbose = True
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=verbose_value,
        memory=memory,
    )

    out = agent.run(OBJECTIVE)
    print(out)

    
