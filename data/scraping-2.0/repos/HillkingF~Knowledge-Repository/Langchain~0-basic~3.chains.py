'''
@Author: 冯文霓
@Date: 2023/6/6
@Purpose: 使用chain链来传入格式化prompt
'''

from langchain.prompts import PromptTemplate  # 格式化提示模板
from langchain.chains import LLMChain     # LLM链
from langchain.llms import OpenAI
from Langchain.units import *     #从根目录开始导入




# 1.创建格式化接收用户输入的prompt提示
prompt = PromptTemplate(
    input_variables=["product"],
    template="what is a good name for a company that makes {product}?"
)


# 2.创建llm原语，即API对象
llm = OpenAI(temperature=0.9)


# 3.创建LLM链,将API对象和prompt格式化输入加入链中（可以理解为一个容器？）
chain = LLMChain(llm=llm, prompt=prompt)

# 4.最后运行链，自动获取结果（需要传入用户参数作为prompt格式化提示的输入）
print(chain.run("socks"))

# 回答：CozyKicks.


# 总结，使用chain就不用使用prompt.format(product="milks") 来格式化prompt了