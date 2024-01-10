from horde_client import HordeClientLLM, TextGenParams

from langchain import LLMChain
from langchain.prompts import PromptTemplate

# Prepare HordeClientLLM
params = TextGenParams(
    max_context_length=256,
    max_length=64,
    temperature=0.8
)

llm = HordeClientLLM(
    # To access public Horde Client
    insecure=True,

    # TextGen Parameters for HordeAI Client
    params=params
)

# Prompt Template
template = """### Instruction:
Create a fancy company name for a company that makes {product}.
### Response:
"""

prompt= PromptTemplate(input_variables=["product"], template=template)

# Chain Prompt with LLM
chain = LLMChain(
    llm = llm,
    prompt = prompt
)

print(chain.run("colorful socks"))

print(chain.run("mobiles"))

