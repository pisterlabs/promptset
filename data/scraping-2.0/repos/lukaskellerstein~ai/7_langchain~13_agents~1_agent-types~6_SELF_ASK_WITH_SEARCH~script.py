from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# First Agent
# ---------------------------
llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

# ZERO_SHOT_REACT_DESCRIPTION
# STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# CONVERSATIONAL_REACT_DESCRIPTION
# CHAT_CONVERSATIONAL_REACT_DESCRIPTION
# REACT_DOCSTORE
# SELF_ASK_WITH_SEARCH <---------------------
self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
result = self_ask_with_search.run(
    "What is the hometown of the reigning men's U.S. Open champion?"
)

print(result)
