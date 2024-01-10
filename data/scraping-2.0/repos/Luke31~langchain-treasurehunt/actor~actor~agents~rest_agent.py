from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI

from actor.third_parties import apidocs


def run_rest_agent(command:str) -> AgentExecutor:
    model = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model, temperature=0)
    tools = load_tools(["requests_all"])

    api_docs = apidocs.get_docs()
    prefix = f"""
            Execute only one request given following api, stop execution and return the response as the answer:
            ----
            {api_docs}
            ----
            
            You are given one command.
            Call following API only using GET query-strings accordingly.

            - Available commands (choose only one): START UP DOWN LEFT RIGHT PICK_UP
            
            Return the response as a result. 
            REMEMBER TO CONSIDER THE FIRST RESPONSE THE ANSWER. NO FURTHER ACTION IS REQUIRED.
            
            COMMAND:
            """
    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             handle_parsing_errors=True,
                             verbose=True,
                             max_execution_time=90,
                             max_iterations=30)
    return agent.run(f"{prefix}\n{command}")
