from dotenv import load_dotenv, find_dotenv
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# Tools
# ---------------------------
search = SerpAPIWrapper()

# SINGLE ACTION
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]


# ---------------------------
# Prompt
# ---------------------------
prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)

# ---------------------------
# LLM
# ---------------------------
llm = OpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# ---------------------------
# Custom Agent = ZERO SHOT
# ---------------------------
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# Now let's test it out!
result = agent_executor.run("How many people live in canada as of 2023?")
print(result)

result = agent_executor.run("how about in mexico?")
print(result)

result = agent_executor.run(
    "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"
)
print(result)
