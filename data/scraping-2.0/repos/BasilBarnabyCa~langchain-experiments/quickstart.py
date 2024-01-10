from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain
from langchain.utilities import SerpAPIWrapper
import os

# Load Environment Variables
load_dotenv(find_dotenv())

# Get API Key
api_key = os.environ["OPENAI_API_KEY"]

# --------------------------------------------------------------------------
# LLMs: Get predictions from OpenAI's LLMs
# --------------------------------------------------------------------------

llm = OpenAI(model_name="text-davinci-003")
# llm = OpenAI(temperature=0.6)
prompt = "What can I do on Canada day?"
print(llm(prompt))
