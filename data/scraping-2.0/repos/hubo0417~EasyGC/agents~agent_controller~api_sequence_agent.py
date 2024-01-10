import re
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseMultiActionAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from agents.base_tools.base_tools import functional_Tool
from configs.prompt_config import SEQUENCE_EXECUTE_API_TOOLS_PROMPT_TEMPLATE


# 创建Controller来识别用户意图以及实现用户意图需要调用的工具集合
class API_Sequence_Agent(BaseMultiActionAgent):
    tools: List[functional_Tool]
    llm: BaseLanguageModel
    intent_template: str = SEQUENCE_EXECUTE_API_TOOLS_PROMPT_TEMPLATE
    prompt = PromptTemplate.from_template(intent_template)
    llm_chain: LLMChain = None

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def output_parser(self, text: str):
        if not text:
            raise ValueError("未能获取到需要解析的文本信息")
        # 使用正则表达式匹配方括号中的内容
        matches = re.findall(r'\[(.*?)\]', text)
        # 将匹配到的内容构造成一个数组
        result_array = []
        if matches:
            result_array = [match.strip() for match in matches[0].split('，')]
        return result_array

    def check_selected_tools(self, tools: list, selected_tools: list) -> bool:
        if not selected_tools or len(selected_tools) <= 0:
            return False
        for select_tool in selected_tools:
            if select_tool not in tools:
                return False
        return True

    # 根据提示(prompt)选择工具
    def choose_tools(self, query) -> List[str]:
        self.get_llm_chain()
        tool_infos = [{tool.name: tool.description} for tool in self.tools]
        resp = self.llm_chain.predict(intents=tool_infos, query=query)
        select_tools = self.output_parser(resp)
        tool_names = [tool.name for tool in self.tools]
        if self.check_selected_tools(tool_names, select_tools) is False:
            return None
        return select_tools

    @property
    def input_keys(self):
        return ["input"]

    # 通过 AgentAction 调用选择的工具，工具的输入是 "input"
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]],
             **kwargs: Any) -> Union[List[AgentAction], AgentFinish]:
        # 单工具调用
        tools = self.choose_tools(kwargs["input"])
        if tools is None:
            return AgentFinish({"output": "无工具"}, log="选择工具时出现不能匹配的情况")
        for tool in self.tools:
            if tool.name == tools[-1]:
                tool.return_direct = True
        result: List[Union[AgentAction, AgentFinish]] = []
        for tool in tools:
            result.append(
                AgentAction(tool=tool, tool_input=kwargs["input"], log=""))
        return result

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]],
                    **kwargs: Any) -> Union[List[AgentAction], AgentFinish]:
        raise NotImplementedError("IntentAgent does not support async")
