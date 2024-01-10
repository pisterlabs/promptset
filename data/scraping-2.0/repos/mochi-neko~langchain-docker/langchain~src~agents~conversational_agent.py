import langchain.llms
from langchain import GoogleSearchAPIWrapper, LLMChain
from langchain.agents import initialize_agent, AgentType, Tool, ZeroShotAgent, AgentExecutor
from langchain.schema import BaseMemory


def setup_agent(llm: langchain.llms.BaseLLM, memory: BaseMemory):
    search = GoogleSearchAPIWrapper()

    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="Useful for when you need to answer questions about current events, the current state of the world or what you don't know."
        ),
    ]

    prefix = """Answer the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin! Use lots of tools, and please answer finally in Japanese.

        Question: {input}
        {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt)
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        tools=tools,
        memory=memory)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True)

    return agent_executor
