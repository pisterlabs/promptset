from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import os
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.environ["SERPAPI_API_KEY"] = constants.SERPKEY

llm = OpenAI(temperature = 0.2)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("Who is Nguyen Phu Trong? What is the value of his age subtract to ten?")