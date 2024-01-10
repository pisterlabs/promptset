import os
import requests
import json
import logging
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function

load_dotenv()

# Agent Wallet service API key to access other agents
# Register on agentwallet.ai/login to get API key
AGENTWALLET_API_KEY = os.getenv("AGENTWALLET_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Tool that uses Agent Wallet proxy to request for other agent
# You'll find other agents to call out from Agent Wallet dashboard
@tool
def get_joke_agent(query: str) -> str:
  """Returns joke agent's response which is a joke on the given topic."""
  logger.info("Calling joke agent now...")
  url = "https://api.agentwallet.ai/agents/AgentOne/chat/invoke"
  payload = json.dumps({"input": {"topic": query}})
  headers = {
      'Authorization': "Bearer " + AGENTWALLET_API_KEY,
      'Content-Type': 'application/json'
  }
  try:
    response = requests.request("POST", url, headers=headers,
                                data=payload).json()
    logger.info("Sucessfully called joke agent.")
    return response["output"]["content"]
  except Exception as e:
    return f"Error: {e}"


# Agent configuration using LangChain
tools = [get_joke_agent]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful personal speech writing assistant. Be concise."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI()

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = ({
    "input":
    lambda x: x["input"],
    "agent_scratchpad":
    lambda x: format_to_openai_functions(x["intermediate_steps"]),
}
         | prompt
         | llm_with_tools
         | OpenAIFunctionsAgentOutputParser())

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Please help me write a speech about AI, but include a joke in the speech."})