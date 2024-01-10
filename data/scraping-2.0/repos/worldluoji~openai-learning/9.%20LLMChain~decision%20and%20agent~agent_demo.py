
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期：2023-01-01；预计送达时间：2023-01-10"

def recommend_product(input: str) -> str:
    return "红色连衣裙"

def faq(intput: str) -> str:
    return "7天无理由退货"

# 每一个 Tool 的对象，在函数之外，还定义了一个名字，并且定义了 Tool 的 description。这个 description 就是告诉 AI，这个 Tool 是干什么用的。AI 会根据问题以及这些描述来做选择题。
tools = [
    Tool(
        name = "Search Order",func=search_order, 
        description="useful for when you need to answer questions about customers orders"
    ),
    Tool(name="Recommend Product", func=recommend_product, 
         description="useful for when you need to answer questions about product recommendations"
    ),
    Tool(name="FAQ", func=faq,
         description="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."
    )
]

# Agent 是有自主行动能力的，它可以根据你提出的要求，直接使用提供的工具采取行动。它不只是做完选择题就完事儿了，而是直接拿起选中的工具进行下一步的行动。
# 在 agent 的类型这里，我们选择了 zero-shot-react-description。这里的 zero-shot 就是指我们在课程一开始就讲过的“零样本分类”，也就是不给 AI 任何例子，直接让它根据自己的推理能力来做决策
# 而 react description，指的是根据你对于 Tool 的描述（description）进行推理（Reasoning）并采取行动（Action）
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
result = agent.run(question)
print(result)


question = "请问你们的货，能送到三亚吗？大概需要几天？"
result = agent.run(question)
print(result)

# 其实它就是把一系列的工具名称和对应的描述交给了 OpenAI，让它根据用户输入的需求，选取对应的工具，然后提取用户输入中和用户相关的信息。本质上，只是我们上面让 AI 做选择题的一种扩展而已。

# 如果你希望 AI 不要不断重试，也不要强行回答，在觉得不靠谱的时候，试个一两次就停下来。那么，你在创建 Agent 的时候，设置 max_iterations 这个参数就好了
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", max_iterations = 2, verbose=True)
