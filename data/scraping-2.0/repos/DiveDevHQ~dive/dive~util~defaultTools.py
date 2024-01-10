from langchain.agents import load_tools
from langchain.llms import OpenAI
from dive.util.configAPIKey import set_openai_api_key


set_openai_api_key()
llm = OpenAI(temperature=0)
STANDARD_TOOLS = load_tools("llm-math", llm = llm)