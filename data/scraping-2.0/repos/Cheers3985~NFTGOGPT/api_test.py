from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import requests
# 定义大语言模型

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_PROXY"] = "http://127.0.0.1:10809"
llm = OpenAI(temperature=0)
# 初始化搜索链和计算链
# search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)


# Define the prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Which NFT scene does your question relate to {question}? (e.g. gaming, art, collectibles)",
)

# Define the chains for each NFT scene
# gaming_chain = LLMChain(llm=llm, prompt=prompt)
# art_chain = LLMChain(llm=llm, prompt=PromptTemplate("Art API interface prompt"))
# collectibles_chain = LLMChain(llm=llm, prompt=PromptTemplate("Collectibles API interface prompt"))

# Define the agent to dynamically call the appropriate chain based on the user's input
from langchain.agents import ZeroShotAgent

agent = ZeroShotAgent(
    llm=llm,
   prompt=prompt,
)

# Run the agent with the user's question
question = "What are the most popular NFT games?"
response = agent(question)

# The agent will prompt the user to select the NFT scene, and then call the appropriate API interface to return the content
print(response)
# 创建一个功能列表，指明这个 agent 里面都有哪些可用工具，agent 执行过程可以看必知概念里的 Agent 那张图
# tools = [
#     Tool(
#         name = "Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events"
#     ),
#     Tool(
#         name="Calculator",
#         func=llm_math_chain.run,
#         description="useful for when you need to answer questions about math"
#     )
# ]
#
# # 初始化 agent
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
#
# # 执行 agent
# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
