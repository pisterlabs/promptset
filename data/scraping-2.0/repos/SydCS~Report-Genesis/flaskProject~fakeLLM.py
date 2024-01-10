import re
import sys
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional


class FakeLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        print("question:", prompt)
        pattern = re.compile(r"^.*(\d+[*/+-]\d+).*$")
        match = pattern.search(prompt)
        if match:
            result = eval(match.group(1))
        elif "？" in prompt:
            rep_args = {"我": "你", "你": "我", "吗": "", "？": "！"}
            result = [(rep_args[c] if c in rep_args else c) for c in list(prompt)]
            result = "".join(result)
        else:
            # result = "很抱歉，请换一种问法。比如：1+1等于几"
            result = "SELECT * from Customers"
        return result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


if __name__ == '__main__':
    message = sys.argv[1]
    llm = FakeLLM()
    print("Answer:", llm(message))
