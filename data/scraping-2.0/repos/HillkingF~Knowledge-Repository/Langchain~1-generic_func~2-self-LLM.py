'''
@Author: 冯文霓
@Date: 2023/6/7
@Purpose: 本节介绍如何自定义LLM包装器。必须实现_call() 。可选则性实现 _identifying_params()
'''

from Langchain.units import *
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

# 自定义一个LLM类，使用LLM作为基类，新类继承LLM类的全部属性和方法
class CustomLLM(LLM):
    # 定义一个属性，注解说明这个属性的类型
    n: int

    @property                # 这个装饰器decorator，将普通的方法转换成属性访问
    def _llm_type(self) -> str:  # _方法名()，这种方法是内部方法，只能在类内部使用，不能供外部用户调用
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.n]  # 返回字符串的0-n-1个字符

    @property
    def _identifying_params(self) -> Mapping[str, Any]:  # 内部方法，不能由类外部用户调用。继承了LLM的基类BaseLLM的方法
        """Get the identifying parameters."""
        return {"n": self.n}



llm = CustomLLM(n=10)
print(llm("This is a foobar thing"))
