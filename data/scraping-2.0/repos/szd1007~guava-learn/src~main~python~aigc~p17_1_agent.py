from langchain.memory import ConversationSummaryMemory
import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import ConversationChain
from  langchain.chains import SequentialChain
from langchain.chains import LLMRequestsChain
openai.api_key = os.environ.get("OPENAI_API_KEY")

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)


def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期: 2023-01-01;预计送达时间：2023-01-01"


def recommend_product(input: str) -> str:
    return "红色连衣裙"


def faq(input: str) -> str:
    return "7天无理由退货"

tools = [
    Tool(
        name="Search Order", func=search_order,
        description="useful for when you need to answer questions about customers orders"
    ),
    Tool(
        name="Recommend Product", func=recommend_product,
        description="useful for when you need to answer questions about product recommendations"
    ),
    Tool(
        name="FAQ", func=faq,
        description="useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."
    )
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

#question = "我想买一件衣服， 但是不知道哪个款式好看， 你能帮我推荐一下吗？"
question = "我有一张订单，订单号是2022adfe，一直没有收到，能麻烦帮我查一下么"

result = agent.run(question)
print(result)