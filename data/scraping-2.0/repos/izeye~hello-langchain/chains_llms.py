from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt_output = prompt.format(product="colorful socks")
print(prompt_output)

chain = LLMChain(llm=llm, prompt=prompt)
chain_output = chain.run("colorful socks")
print(chain_output)
