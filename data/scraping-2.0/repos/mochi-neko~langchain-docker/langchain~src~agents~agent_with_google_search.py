from langchain.agents import ZeroShotAgent, AgentExecutor, load_tools
import langchain.llms
from langchain.schema import BaseMemory
from langchain import LLMChain

def setup_agent(llm : langchain.llms.BaseLLM, memory: BaseMemory):
    tools = load_tools(["google-search"], llm=llm)
    prefix = """次の質問にできる限り答えてください。次のツールにアクセスできます:"""
    suffix = """始めましょう!

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True)

    return agent_executor
