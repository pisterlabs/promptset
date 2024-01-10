from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# First Agent
# ---------------------------
docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]

llm = OpenAI(temperature=0, model_name="text-davinci-002")


# ZERO_SHOT_REACT_DESCRIPTION
# STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# CONVERSATIONAL_REACT_DESCRIPTION
# CHAT_CONVERSATIONAL_REACT_DESCRIPTION
# REACT_DOCSTORE <---------------------
# SELF_ASK_WITH_SEARCH
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
result = react.run(question)
print(result)
