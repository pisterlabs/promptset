from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


# 简单的Demo示例
def generate_llm_reply():
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("Who is Xijinpin's girlfriend? What is her current age raised to the 2 power?")


if __name__ == '__main__':
    generate_llm_reply()
