from langchain.agents import AgentExecutor

def get_executor_from_agent_and_tools(agent, tools, verbose=False, agent_kwargs={}):
    """
    Get an executor from an agent and a list of tools.

    :param agent: The agent to create the executor from.
    :type agent: Agent

    :param tools: The list of tools to include in the executor.
    :type tools: List[Tool]

    :param verbose: Whether to enable verbose output. Defaults to False.
    :type verbose: bool

    :return: The executor created from the agent and tools.
    :rtype: AgentExecutor
    """
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose, agent_kwargs=agent_kwargs
    )
