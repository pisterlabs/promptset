import re
import json
from typing import Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser


class OutputParserForAgent(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # 解析action和action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        # 检查是否应该停止
        if action == "Final Response":
            return AgentFinish(
                return_values={"output": action_input},
                log=llm_output,
            )
        else:
            # 定义正则表达式模式，匹配包含键值对的花括号结构
            pattern = r'\{[^{}]*\}'
            # 使用正则表达式进行匹配
            action_input = re.findall(pattern, action_input)[0]
            action_input = json.loads(action_input)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)
    
      
class OutputParserForRoot(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # 解析action和action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        # 检查是否应该停止
        if action == "Final Response":
            response = action_input
            return AgentFinish(
                return_values={"output": response},
                log=llm_output,
            )
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)