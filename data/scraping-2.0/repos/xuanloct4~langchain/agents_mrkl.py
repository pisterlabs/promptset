
import environment
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding
from agents_tools import search_tool_serpapi, chinook_db_tool, calculator_tool

from langchain.agents import initialize_agent
from langchain.agents import AgentType

# llm = ChatOpenAI(temperature=0)
# llm1 = OpenAI(temperature=0)
toolLLM=llm
tools = [
    search_tool_serpapi(),
    calculator_tool(toolLLM),
    chinook_db_tool(toolLLM)
]

# from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
# from langchain.agents import initialize_agent, Tool
# from langchain.agents import AgentType
# search = SerpAPIWrapper()
# llm_math_chain = LLMMathChain(llm=llm, verbose=True)
# db = SQLDatabase.from_uri("sqlite:///./Chinook.db")
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
# tools = [
#     Tool(
#         name = "Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events. You should ask targeted questions"
#     ),
#     Tool(
#         name="Calculator",
#         func=llm_math_chain.run,
#         description="useful for when you need to answer questions about math"
#     ),
#     Tool(
#         name="FooBar DB",
#         func=db_chain.run,
#         description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context"
#     )
# ]


mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(mrkl.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"))
print(mrkl.run("What is the full name of the artist who recently released an album called 'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs are in the FooBar database?"))
