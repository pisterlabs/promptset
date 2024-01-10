from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0, openai_api_key=api_key)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("Who is the current leader of Japan? What is the largest prime number that is smaller than their age?")