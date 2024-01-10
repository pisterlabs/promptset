from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List


class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    prompt_1 = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
#     chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = PromptTemplate(
        input_variables=["product"],
        template="What is a good slogan for a company that makes {product}?",
    )
#    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    def concatenate(self, input_1: str, input_2: str) -> str:
        # concat_output = concat_chain.run("colorful socks")
        concat_output = concat_chain.run("colorful socks")
        return f"Concatenated output:\n{concat_output}"

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}

