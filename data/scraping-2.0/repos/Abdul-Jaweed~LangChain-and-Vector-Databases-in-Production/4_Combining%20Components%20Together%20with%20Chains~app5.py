## Custom Chain

# The LangChain library has several predefined chains for different applications like Transformation Chain, LLMCheckerChain, LLMSummarizationCheckerChain, and OpenAPI Chain, which all share the same characteristics mentioned in previous sections. It is also possible to define your chain for any custom task. In this section, we will create a chain that returns a word's meaning and then suggests a replacement.

# It starts by defining a class that inherits most of its functionalities from the Chain class. Then, the following three methods must be declared depending on the use case. The input_keys and output_keys methods let the model know what it should expect, and a _call method runs each chain and merges their outputs.



from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from typing import Dict, List

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)


class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain
    
    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)
    
    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str,str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}
    


prompt_1 = PromptTemplate(
    input_variables=["word"],
    template="What is the meaning of the following word '{word}'?",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["word"],
    template="What is a word to replace the following: {word}?",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
concat_output = concat_chain.run("artificial")
print(f"Concatenated output:\n{concat_output}")