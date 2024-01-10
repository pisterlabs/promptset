from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

from dynamic import dynamic
from dynamic.classes.dynamic_agent  import DynamicAgent


llm = OpenAI(temperature=0, streaming=True, verbose=True)

tool_list = ["google-serper"]

inline_agent = initialize_agent(
    tools=load_tools(tool_list), llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
)

streaming_agent = DynamicAgent(
    tool_list=tool_list, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
)

@dynamic(streaming=True)
def agent():
    return streaming_agent