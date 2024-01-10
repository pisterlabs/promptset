# langchan research agent
import openai
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.utilities import ArxivAPIWrapper

# Set up the OpenAI API
openai.api_key = "sk-" # Replace the string content with your OpenAI API key

llm = ChatOpenAI(temperature=0) # Initialize the LLM to be used

arxiv = ArxivAPIWrapper()
arxiv_tool = Tool(
    name="arxiv_search",
    description="Search on arxiv. The tool can search a keyword on arxiv for the top papers. It will return publishing date, title, authors, and summary of the papers.",
    func=arxiv.run
)

tools = [arxiv_tool]

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.run("What is ReAct reasoning and acting in language models?")


