import re
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from configs.prompt_config import SINGLE_EXECUTE_API_TOOLS_PROMPT_TEMPLATE
from agents.base_tools.base_tools import functional_Tool


# 创建Controller来识别用户意图并分发给对应的工具
class API_Single_Agent(BaseSingleActionAgent):
    tools: List[functional_Tool]
    llm: BaseLanguageModel
    intent_template: str = SINGLE_EXECUTE_API_TOOLS_PROMPT_TEMPLATE
    prompt = PromptTemplate.from_template(intent_template)
    llm_chain: LLMChain = None

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def output_parser(self, text: str):
        if not text:
            raise ValueError("未能获取到需要解析的文本数据")
        matches = re.findall(r'\[(.*?)\]', text)
        # 将匹配到的内容构造成一个数组
        result_array = []
        if matches:
            result_array = [match.strip() for match in matches[0].split('，')]
        return result_array

    # 根据提示(prompt)选择工具
    def choose_tools(self, query) -> List[str]:
        self.get_llm_chain()
        tool_names = [tool.name for tool in self.tools]
        resp = self.llm_chain.predict(intents=tool_names, query=query)
        select_tools = [(name, resp.index(name)) for name in tool_names
                        if name in resp]
        select_tools.sort(key=lambda x: x[1])
        return [x[0] for x in select_tools]

    @property
    def input_keys(self):
        return ["input"]

    # 通过 AgentAction 调用选择的工具，工具的输入是 "input"
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]],
             **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        # 单工具调用
        tool_name = self.choose_tools(kwargs["input"])[0]
        return AgentAction(tool=tool_name, tool_input=kwargs["input"], log="")

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]],
                    **kwargs: Any) -> Union[List[AgentAction], AgentFinish]:
        raise NotImplementedError("IntentAgent does not support async")
