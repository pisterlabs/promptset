import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from repolya.local.vllm import get_vllm_llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


_model = "/home/songz/text-generation-webui/models/TheBloke_SUS-Chat-34B-AWQ"

llm2 = get_vllm_llm(_model,
                    0.01,
                    0.9,
                    200,
                    [""]
)


template = """Question: {question}

Answer: Let's think step by step."""

# template = """### Human:
# {question}
# Let's think step by step.

# ### Assistant:
# """

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm2)

question = "Who was the US president in the year the first Pokemon game was released?"
print(llm_chain.run(question))

