from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.tools import AIPluginTool
from langchain.callbacks import get_openai_callback
from langchain.tools.requests.tool import RequestsPostTool
from langchain.requests import RequestsWrapper

tool = AIPluginTool.from_plugin_url("http://127.0.0.1:8000/.well-known/ai-plugin.json")

llm = ChatOpenAI(temperature=0)
tools = load_tools(["requests"] )
tools.extend([tool, RequestsPostTool(requests_wrapper=RequestsWrapper())])

agent_chain = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# agent_chain.run("Retrieve my Wealthsimple Trade account status using the API on http://127.0.0.1:8000")

with get_openai_callback() as cb:
    agent_chain.run("Make a deposit of 1 dollar to my account on http://127.0.0.1:8000/make_deposit from the WSTrade plugin")