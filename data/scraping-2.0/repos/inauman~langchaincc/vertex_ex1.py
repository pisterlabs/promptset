#!pip install google-cloud-aiplatform
#import Google API key
from apikey import GOOGLE_API_KEY

from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
from langchain.utilities import Portkey

headers = Portkey.Config(
    api_key=GOOGLE_API_KEY
)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = VertexAI(headers=headers)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

response = llm_chain.run(question)

print(response)