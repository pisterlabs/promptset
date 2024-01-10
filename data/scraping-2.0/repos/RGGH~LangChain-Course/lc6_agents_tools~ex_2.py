from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain import LLMMathChain

# pip install google-search-results

memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI(temperature=0)

#llm_math_chain = LLMMathChain(llm=llm) deprecated !!
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper()

tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about recent or current events or the current state of the world",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math or calulate an average",
        return_direct=True,
    ),
]

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What was the average age of the UK prime minister between 2019 and 2023?")
