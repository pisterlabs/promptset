from dotenv import load_dotenv
load_dotenv()

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

chat = ChatOpenAI(temperature=0)

llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)


agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is 50 * 234 + 234 / 123?")
