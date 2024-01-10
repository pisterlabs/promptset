from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

temp=prompt
print(temp)
# result=llm_chain.run(question)
# print(result)