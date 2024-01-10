from langchain.llms import OpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

OPENAI_API = "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"

llm = OpenAI(model_name="text-davinci-003", openai_api_key=OPENAI_API)

template = """
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines
"""
prompt = PromptTemplate(
  input_variables=['concept'],
  template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("autoencoder"))

second_prompt = PromptTemplate(
  input_variables=["ml_concept"],
  template="Turn the concept description of {ml_concept} and explain it to me like I'm five",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
explanation = overall_chain.run("autoencoder")

print(explanation)
