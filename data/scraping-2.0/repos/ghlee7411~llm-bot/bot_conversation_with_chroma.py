from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from tools import mermaid, search, recipe_search_engine
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)
del load_dotenv

# for debugging
import langchain
langchain.debug = False


def main():
    llm = ChatOpenAI(temperature=0)
    tools = [
        Tool(
            name="mermaid_diagram_generator",
            func=mermaid,
            description="You can depict a situation using a Mermaid diagram, illustrating the scenario in visual form.",
        ),
        Tool(
            name="recipe_database_search_engine",
            func=recipe_search_engine,
            description="You can ask any cooking question and get the insight.",
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm=OpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    
    


if __name__ == "__main__":
    main()