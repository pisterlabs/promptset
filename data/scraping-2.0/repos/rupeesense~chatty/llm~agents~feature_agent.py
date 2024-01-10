from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

from context_store.store import ContextStore
from llm.agents.tools import SavingsTool, FactualTool

if __name__ == '__main__':
    ContextStore().connect()

    tools = [SavingsTool(), FactualTool()]

    agent_executor = initialize_agent(
        tools=tools,
        llm=OpenAI(temperature=0,
                   openai_api_key='sk-'),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    SYS_PROMPT = ''' 
    You are the best personal finance assistant. You will have context of user financial data.
    When answering questions, you should try explain the numbers in a very intuitive manner.
    You are given a set of tools to help you answer the query, but only use them if necessary.
    Report all numbers in rupees.
    Today's date is 19th Sept 2023.
    
    '''

    # agent_executor.run(
    #     "How have my monthly savings varied over the past year? Report all numbers in rupees. Try to explain the numbers "
    #     "in a helpful manner. Today's date is 19th Sept 2023. My user_id is: user1."
    # )

    # agent_executor.run(
    #     "What is my total savings across all bank accounts? Report all numbers in rupees. Try to explain the numbers "
    #     "in a helpful manner. Today's date is 19th Sept 2023. My user_id is: user1."
    # )
    #
    # agent_executor.run(
    #     "How much interest have I earned on my savings over the past year for accountA? Report all numbers in rupees. Try to explain the numbers "
    #     "in a helpful manner. Today's date is 19th Sept 2023. My user_id is: user1. Interest is compounded annually."
    # )

    # agent_executor.run(
    #     "How does my savings rate compare to the recommended savings rate? Report all numbers in rupees. Try to explain the numbers "
    #     "in a helpful manner. Today's date is 19th Sept 2023. My user_id is: user1. Interest is compounded annually."
    # )

    # agent_executor.run(
    #     SYS_PROMPT + "How have my monthly savings varied over the past year?"
    # )

    agent_executor.run(
        SYS_PROMPT + "What has been my highest and lowest savings month so far this year?"
    )

    # agent_executor.run(
    #     "What percentage of my income am I currently saving for all my accounts? Report all numbers in rupees. Try to explain the numbers "
    #     "in a helpful manner. Today's date is 29th Sept 2023. My user_id is: user1."
    # )
