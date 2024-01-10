from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool

#tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
tool = AIPluginTool.from_plugin_url("https://www.instacart.com/.well-known/ai-plugin.json")

llm = chat = ChatOpenAI(temperature=0, openai_api_key="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")
tools = load_tools(["requests_all"])
tools += [tool]

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

#agent_chain.run("what t shirts are available in klarna?")
agent_chain.run("Make a shopping cart with these ingredients: steak, rosemary, butter, potato")