from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains import LLMChain, LLMMathChain
from langchain_experimental.utilities import PythonREPL
from chatglm3_6b_llm import Chatglm3_6b_LLM
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

#指定ChatGLM2-6B的API endpoint url，用langchain的ChatOpanAI类初始化一个ChatGLM的chat模型
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
        model_name="chatglm",
        openai_api_base="http://127.0.0.1:6006/v1",
        openai_api_key="1234",
        streaming=False,
    )

#使用会话实体内存，利用ChatGLM在会话过程中分析提到的实体(Entity)
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
entity_memory = ConversationEntityMemory(llm=llm, k=5 )

#生成会话链
# from langchain.chains import ConversationChain
# conversation = ConversationChain(
#             llm=llm, 
#             prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
#             memory=entity_memory,
#             verbose=True,
#         ) 


class WeatherInput(BaseModel):
    location: str = Field(description="The name of the city to be queried")

@tool("get_weather_tool", return_direct=True, args_schema=WeatherInput)
def get_weather_tool(location: str) -> str:
    """get_weathe_tool the current weather for `city_name`"""
    print(f"get_weather_tool the current weather for:{location}")
    return f"城市:{location}，天气：多云 23°C "


tools = [get_weather_tool]

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
#                          handle_parsing_errors=True, max_iterations=3, early_stopping_method="generate")

# user_input = "杭州的天气如何？"
# answer = agent.run(user_input)
# print(answer)

agent_prompt = """Answer the following questions as best you can. You have access to the following tools:

Search Order:
一个能够查询订单信息，获得最新的订单情况的工具，参数是输入订单id
Recommend product: 一个能够基于商品及用户
信息为用户进行商品推荐导购的工具，参数是输入要推荐的商品类型

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: Search Order, Recommend product

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

\`\`\`
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
\`\`\`

ALWAYS use the following format:

Question: the input question you
must answer
Thought: you should always think about what to do
Action:
\`\`\`
$JSON_BLOB
\`\`\`
Observation:
the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the
final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.'
{question}
"""

question="我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗"
messages = [
    HumanMessagePromptTemplate.from_template(
        template=agent_prompt,
    ).format(question=question),
]

# 模拟电商订单
def search_order(input: str)->str:
    print("调用search_order：一个能够查询订单信息，获得最新的订单情况的工具:")
    return "{order}，订单状态：已发货".format(order=input)

# 模拟商品推荐
def recommend_product(input: str)->str:
    print("调用recommend_product：一个能够基于商品及用户信息为用户进行商品推荐导购的工具:")
    return "黑色连衣裙"
tools = [
    Tool(
        name="Search Order",
        func=search_order,
        description="""一个能够查询订单信息，获得最新的订单情况的工具，参数是输入订单id"""
    ),
    Tool(
        name="Recommend product",
        func=recommend_product,
        description="一个能够基于商品及用户信息为用户进行商品推荐导购的工具，参数是输入要推荐的商品类型"
    )
]

agent_tools = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True)
result = agent_tools.run("我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗")
print(result)
