from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import OpenAI
import os
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool


def chat():
    model = AzureChatOpenAI(
        model_name="gpt-35-turbo",
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_base=os.environ["AZURE_OPENAI_API_BASE"],
        openai_api_version="2023-06-01-preview",
        deployment_name=os.environ["AZURE_OPENAI_MODEL"],
        temperature=0,
    )

    search = DuckDuckGoSearchRun()

    duckduckgosearch = Tool(
        name="duckduckgo-search",
        func=search.run,
        description="useful for when you need to search for latest information in web",
    )

    tools = load_tools(["serpapi", "llm-math"], llm=model)

    agent = initialize_agent(
        tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.verbose = True
    output = agent.run([HumanMessage(content="今の名古屋の最高気温の摂氏温度の値の2乗の値は？")])

    print(output)
