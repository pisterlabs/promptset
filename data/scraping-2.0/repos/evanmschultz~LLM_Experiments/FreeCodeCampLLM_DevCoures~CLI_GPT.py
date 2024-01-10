from langchain.tools import ShellTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType, AgentExecutor
import os

shell_tool = ShellTool()

llm = ChatOpenAI(temperature=0, model="gpt-4")

shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")
agent: AgentExecutor = initialize_agent(
    tools=[shell_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run(
    "Create a text file called empty and inside it, add code that trains a basic convolutional neural network for 4 epochs"
)
