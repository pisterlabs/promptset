from http import HTTPStatus
import dashscope
from typing import Optional, List, Dict, Mapping, Any
from langchain.llms.base import LLM
class Dashscope(LLM):
    '''
    根据源码解析在通过LLMS包装的时候主要重构两个部分的代码
    _call 模型调用主要逻辑,输入问题，输出模型相应结果
    _identifying_params 返回模型描述信息，通常返回一个字典，字典中包括模型的主要参数
    '''
    dashscope.api_key = "sk-97d523aa76184c338c80b32954643e40"


    @property
    def _llm_type(self) -> str:
        # 模型简介
        return "Dashscope"



    def _post(self, prompt):
        # 模型请求响应

        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_max,
            prompt=prompt
        )
        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        content = ""
        if response.status_code == HTTPStatus.OK:
            content = response.output.text
            print(response.output)  # The output text
            print(response.usage)  # The usage information
        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.
        return content

    def _call(self, prompt: str,
              stop: Optional[List[str]] = None) -> str:
        # 启动关键的函数
        print(prompt)
        content = self._post(prompt)
        return content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters.
        """
        _param_dict = {
        }
        return _param_dict
if __name__ == '__main__':
    llm = Dashscope()
    result = llm("邮政董事长是谁")
    print(result)

