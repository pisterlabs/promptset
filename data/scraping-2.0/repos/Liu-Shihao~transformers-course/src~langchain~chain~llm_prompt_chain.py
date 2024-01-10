from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
"""
最核心的链类型是 LLMChain，它由 PromptTemplate 和 LLM 组成。
"""

llm = OpenAI(temperature=0.9)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# 创建一个非常简单的链，它将接受用户输入，用它格式化提示，然后将它发送到 LLM
chain = LLMChain(llm=llm, prompt=prompt)

chain.run("colorful socks")
