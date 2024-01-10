from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.tools import BaseTool

from lope_tools import SenseTagTool


class CountSenseChain(Chain):
    chain_1: BaseTool
    chain_2: LLMChain

    @property
    def input_keys(self) -> list[str]:
        return ["text", "target_word"]
    
    @property
    def output_keys(self) -> list[str]:
        return ["output"]
    
    def _call(self, inputs: dict[str, str]) -> dict[str, str]:
        tagged = self.chain_1.run(inputs["text"])
        count = self.chain_2.run(text=tagged, target_word=inputs["target_word"])
        return {'output': count}