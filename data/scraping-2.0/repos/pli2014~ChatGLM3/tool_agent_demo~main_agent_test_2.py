import os
import re
import json
from typing import List, Union

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish

os.environ["OPENAI_API_KEY"] = "1234"

def find_person(name: str):
    """
    模拟本地数据库查询。

    Args:
        name (str): 人物名称，由LLM提取。

    Returns:
        _type_: _description_
    """
    info = {
        '张三': '男',
        '小红': '女'
    }
    return info.get(name, f'未找到{name}的性别信息，我应该直接返回 Observation: 未知')


def recommend_item(gender: str):
    """
    根据人物性别推荐不同的商品。

    Args:
        gender (str): 人物的性别，由 LLM 提取。

    Returns:
        _type_: _description_
    """
    recommend = {
        '男': ['Steam爆款', 'RTX-9090', 'iPhone 80'],
        '女': ['夏季碎花裙', '轻盈帆布包', '琉璃唇釉'],
        '未知': ['AJ新品', '手冲咖啡']
    }
    return recommend.get(gender, f'未找到合适的推荐商品，我应该返回 Final Answer: 随便买些什么吧，只要消费就能快乐！')

tools = [
    Tool(
        name = "查询人物性别",
        func=find_person,
        description="通过人名查找该人物的性别时用的工具，输入应该是人物的名字"
    ),
    Tool(
        name = "根据性别推荐商品",
        func=recommend_item,
        description="当知道了一个人性别后想进一步获得他可能感兴趣的商品时用的工具，输入应该是人物的性别"
    )
]


template_zh = """按照给定的格式回答以下问题。你可以使用下面这些工具：

{tools}

回答时需要遵循以下用---括起来的格式：

---
Question: 我需要回答的问题
Thought: 回答这个上述我需要做些什么
Action: `{tool_names}` 中的其中一个工具名
Action Input: 选择工具所需要的输入
Observation: 选择工具返回的结果
...（这个思考/行动/行动输入/观察可以重复N次）
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案
---

现在开始回答，记得在给出最终答案前多按照指定格式进行一步一步的推理。

Question: {input}
{agent_scratchpad}
"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str           # 标准模板
    tools: List[Tool]       # 可使用工具集合
    
    def format(
            self, 
            **kwargs
        ) -> str:
        """
        按照定义的 template，将需要的值都填写进去。

        Returns:
            str: 填充好后的 template。
        """
        intermediate_steps = kwargs.pop("intermediate_steps")       # 取出中间步骤并进行执行
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts                       # 记录下当前想法
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])    # 枚举所有可使用的工具名+工具描述
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])                        # 枚举所有的工具名称
        cur_prompt = self.template.format(**kwargs)
        print(cur_prompt)
        return cur_prompt


prompt = CustomPromptTemplate(
    template=template_zh,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):
    
    def parse(
            self, 
            llm_output: str
        ) -> Union[AgentAction, AgentFinish]:
        """
        解析 llm 的输出，根据输出文本找到需要执行的决策。

        Args:
            llm_output (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            Union[AgentAction, AgentFinish]: _description_
        """
        if "Final Answer:" in llm_output:       # 如果句子中包含 Final Answer 则代表已经完成
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # 'Thought: 首先我需要知道张三的性别，然后才能根据性别推荐合适的商品。\nAction: 查询人物性别\nInput: 张三\nAction Output: 男\n\n Thought: 既然张三是男性，我会推荐一些男性可能感兴趣的商品。\n
        #  Action: 根据性别推荐商品\nInput: 男\nAction Output: 请稍等，我需要调用相关工具来获取推荐结果'

        print(f"llm_output:{llm_output}")
        regex = r"Thought.*Action:(.*?)\nAction Input:[\s]*(.*?)\n"  # 解析 action_input 和 action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()

#llm = OpenAI(temperature=0)
#指定ChatGLM2-6B的API endpoint url，用langchain的ChatOpanAI类初始化一个ChatGLM的chat模型
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
        model_name="chatglm",
        openai_api_base="http://127.0.0.1:6006/v1",
        openai_api_key="1234",
        streaming=False,
        temperature=0
    )
llm_chain = LLMChain(
    llm=llm, 
    prompt=prompt
)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True
)

res = agent_executor.run(
    "我想送点礼物给小红"
)
print(res)