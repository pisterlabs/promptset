from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent_executor = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent_executor.run(
    "What's the average price of a 1 bedroom condo \
        in New York City in 2022. Calculate a 20%% deposit\
            for it."
)
