#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

'''
以客户的场景，演示agent的用法
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期：2023-01-01；预计送达时间：2023-01-10"

def recommend_product(input: str) -> str:
    return "红色连衣裙"

def faq(intput: str) -> str:
    return "7天无理由退货"

def create_agent():
    llm = OpenAI(temperature=0)

    # 定义了几个Tool，chatgpt会通过描述调用对应的函数func
    tools = [
        Tool(
            name = "Search Order", func=search_order, 
            description="useful for when you need to answer questions about customers orders"
        ),
        Tool(name="Recommend Product", func=recommend_product, 
            description="useful for when you need to answer questions about product recommendations"
        ),
        Tool(name="FAQ", func=faq,
            description="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."
        )
    ]

    # zero-shot-react-description 零样本分类，全部用描述来判断
    # max_iterations 最多迭代3次，如果得不到答案可以进入人工介入
    
    # 框架中的promot也很有意思，大家可以看一下
    # PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
    # FORMAT_INSTRUCTIONS = """Use the following format:
    #
    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original input question"""
    # SUFFIX = """Begin!
    #
    # Question: {input}
    # Thought:{agent_scratchpad}"""
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", max_iterations = 3, verbose=True)
    return agent


if __name__ == '__main__':
    get_api_key()
    agent = create_agent()

    # Agent 每一步的操作，可以分成 5 个步骤，分别是 Action、Action Input、Observation、Thought，最后输出一个 Final Answer。
    # Action，就是根据用户的输入，选择应该选取哪一个 Tool，然后行动。
    # Action Input，就是根据需要使用的 Tool，从用户的输入里提取出相关的内容，可以输入到 Tool 里面。
    # Oberservation，就是观察通过使用 Tool 得到的一个输出结果。
    # Thought，就是再看一眼用户的输入，判断一下该怎么做。如果没有问题就进入下一步，如果答案有问题就回到Action。
    # Final Answer，就是 Thought 在看到 Obersavation 之后，给出的最终输出。
    question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
    result = agent.run(question)
    print(result)

    question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
    result = agent.run(question)
    print(result)

    question = "请问你们的货，能送到三亚吗？大概需要几天？"
    result = agent.run(question)
    print(result)


