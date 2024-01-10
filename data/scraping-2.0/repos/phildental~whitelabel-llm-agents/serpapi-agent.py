import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')
serpapi_api_key = os.environ.get('SERPAPI_API_KEY')


# First, let's load the language model we're going to use to control the agent.

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=serpapi_api_key)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("What can you tell me about Uberlandia, Minas Gerais, Brazil?")
