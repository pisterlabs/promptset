# 使用搜索工具进行自我询问进行推理
# 利用一种叫做 “Follow-up Question（追问）”加“Intermediate Answer（中间答案）”的技巧，来辅助大模型寻找事实性问题的过渡性答案，从而引出最终答案。

from langchain.llms.openai import OpenAI
from langchain.serpapi import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType

llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name='Intermediate Answer',
        func=search.run,
        description="useful for when you need to ask with search",
    ),
]

agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True, )
agent.run("使用玫瑰作为国花的国家的首都是哪里?")
# agent.run('包含小龙女的武侠小说的作者是谁？')
