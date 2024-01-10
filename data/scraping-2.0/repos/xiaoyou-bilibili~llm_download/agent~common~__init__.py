from typing import Callable
from langchain.schema.agent import AgentActionMessageLog, AgentFinish
import json


def return_parse(response: str, default_out: dict) -> Callable:
    def parse(output):
        default_out["output"] = output.content
        # 如果没有函数调用就直接返回给用户默认的输出
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values=default_out, log=output.content)

        # 解析一下函数调用的结果
        function_call = output.additional_kwargs["function_call"]
        name = function_call['name']
        inputs = json.loads(function_call['arguments'])

        # 直接把函数调研的结果给返回即可
        if response == name:
            return AgentFinish(return_values=inputs, log=str(function_call))
        # 否则就返回action
        else:
            return AgentActionMessageLog(tool=name, tool_input=inputs, log="", message_log=[output])

    return parse
