from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools, initialize_agent, AgentType

tools = load_tools(["python_repl"])
responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

agent.run("whats 2 + 2")

# > Entering new AgentExecutor chain...
# Action: Python REPL
# Action Input: print(2 + 2)
# Observation: Python REPL is not a valid tool, try another one.
# Thought:Final Answer: 4

# > Finished chain.