from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool
mine = AIPluginTool
tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
with open('/Users/yonizirman/Documents/GitHub/KnowledgeHouse/api.key', 'r') as file:
    openai_api_key = file.read().strip()
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

tools = load_tools(["requests_all"])
for tool in tools:
    print('\ntool is ', tool)
tools += [tool]

agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
#agent_chain.run("what t shirts are available in klarna?")