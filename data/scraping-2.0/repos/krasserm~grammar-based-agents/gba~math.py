import re
from typing import List, Dict, Any, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate

_PROMPT_TEMPLATE = """[INST] Your task is to create python code to solve a mathematical problem and assign the result to a variable named `result`. Do not perform the calculation yourself no matter how trivial it is, as you are not good at math. Instead you should generate Python code to do the calculation for you. The result of the calculation should be assigned to a variable named `result`. Provide a single answer using python code and wrap your code answer using ```. Ensure that the python code is as simple as possible while still being correct.
Mathematical problem: {question}
[/INST]"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


class Llama2Math(Chain):
    llm: LLM
    prompt: PromptTemplate = PROMPT

    @property
    def input_keys(self) -> List[str]:
        return ["message"]

    @property
    def output_keys(self) -> List[str]:
        return ["result"]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        output = (self.prompt | self.llm).invoke({"question": inputs["message"]})
        code = self._parse_output(output)
        code = code.strip("\n")

        print("Executing Llama2Math Python code:")
        print(f"```\n{code}\n```")

        try:
            loc_variables = {}
            exec(code, globals(), loc_variables)
            result = loc_variables["result"]
            return {"result": f"{result:.5f}"}
        except Exception as e:
            raise ValueError(f"LLM output could not be evaluated (output='{output}', error='{e}')")

    @staticmethod
    def _parse_output(output: str) -> str:
        match = re.search(r"^```(.*)```$", output, re.DOTALL)
        if not match:
            raise ValueError(f"LLM output could not be parsed (output='{output}')")
        return match.group(1)
