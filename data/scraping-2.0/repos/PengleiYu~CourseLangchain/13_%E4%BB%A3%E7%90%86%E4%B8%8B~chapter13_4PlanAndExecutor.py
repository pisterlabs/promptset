# 总分总: 先指定计划再按步骤执行最终汇总结论
from langchain.chat_models.openai import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.serpapi import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain

search = SerpAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name='Search',
        func=search.run,
        description='useful for when you need to answer questions about current events',
    ),
    Tool(
        name='Calculator',
        func=llm_math_chain.run,
        description='useful for when you need to answer questions about math',
    ),
]

model = ChatOpenAI()
planner = load_chat_planner(llm=model)
executor = load_agent_executor(llm=model, tools=tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
# result = agent.run('在北京，100美元能买几束玫瑰?')
agent.run('我想考上清华大学应该怎么做？')
# print(result)
