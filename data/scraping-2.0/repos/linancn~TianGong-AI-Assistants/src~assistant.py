import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.tools import DuckDuckGoSearchRun, E2BDataAnalysisTool
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable

load_dotenv()


# assistant_tools = [{"type": "code_interpreter"}, {"type": "retrieval"}]

assistant_tools = [
    {"type": "retrieval"},
    E2BDataAnalysisTool(),
    DuckDuckGoSearchRun(),
]

langchain_tools = [
    E2BDataAnalysisTool(),
    DuckDuckGoSearchRun(),
]

agent = OpenAIAssistantRunnable.create_assistant(
    name="langchain assistant test",
    instructions="You are a personal math tutor. Write and run code to answer math questions. You can also search the internet.",
    tools=assistant_tools,
    model="gpt-4-1106-preview",
    as_agent=True,
)

agent_executor = AgentExecutor(agent=agent, tools=langchain_tools, verbose=True)

output = agent_executor.invoke(
    {"content": "How is the Dynamic Analysis of Global Copper Flows, search the uploaded files"}
)

# print(output)
