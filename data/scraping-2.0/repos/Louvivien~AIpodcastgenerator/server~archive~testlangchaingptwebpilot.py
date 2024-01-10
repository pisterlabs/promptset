from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the AIPluginTool with the specific plugin URL
tool = AIPluginTool.from_plugin_url("https://webreader.webpilotai.com/openapi.yaml")

llm = ChatOpenAI(temperature=0)

# Load the tools and add the AIPluginTool to the list
tools = load_tools(["requests_all"])
tools += [tool]

memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

print(agent_chain.run("What is this: https://en.wikipedia.org/wiki/Marcia_Reale"))
