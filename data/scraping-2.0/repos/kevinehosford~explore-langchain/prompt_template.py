from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

template = """
You are a narcissistic CEO of a rocket company.

Return a new business idea. It should be likely to fail and extremely provocative.

What is a your business idea for the {industry} industry?
"""

prompt_template = PromptTemplate(input_variables=["industry"], template=template)

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.9, max_tokens=100)

chain = LLMChain(llm=llm, prompt=prompt_template)

industries = ["fast food", "insurance", "online dating"]

for industry in industries:
    print(chain.run(industry=industry))
