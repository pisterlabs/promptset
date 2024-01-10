from langchain.agents import initialize_agent

def initialize_conversational_agent(agent_type, tools, llm_model, memory, system_msg, verbose):
    conversational_agent = initialize_agent(  # Creates conversational agent
                                            agent = agent_type,
                                            tools = tools,
                                            llm = llm_model,
                                            memory = memory,
                                            agent_kwargs = {"system_message": system_msg},
                                            verbose = verbose)
    return conversational_agent
    