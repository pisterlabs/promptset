from langchain.llms import VertexAI
# import your tools here
from tools.tools import get_google_search
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI as OpenAI

from dotenv import load_dotenv
import os
load_dotenv()
if __name__ == "__main__":
    pass

# Define your tools here based on tools.py. You can use the template below.
google_search = Tool(
    name="GoogleSearch",
    func=get_google_search,
    description="useful for when you need get a google search result",
)
def getLLM(temperture):
    llm_type = os.getenv("LLM_TYPE")
    if llm_type == "openai":
        llm = OpenAI(model_name=os.getenv("OPENAI_MODEL"))
    elif llm_type == "vertexai":
        llm = VertexAI(temperature=temperture, verbose=True, max_output_tokens=2047,model_name=os.getenv("VERTEX_MODEL"))
    return llm
# Implement your agents here. You can use the template below.
def agent_template(temperture=0) -> AgentExecutor:
    print("*" * 79)
    print("AGENT: Agent template!")
    print("*" * 79)
    llm = getLLM(temperture)
    tools_for_agent = [
        google_search
    ]

    agent = initialize_agent(
        tools_for_agent,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent