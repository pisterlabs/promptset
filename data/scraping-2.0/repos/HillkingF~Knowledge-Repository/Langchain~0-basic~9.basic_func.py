'''
@Author: 冯文霓
@Date: 2023/6/7
@Purpose: 本节介绍llm类的基本功能(文本生成、多条文本生成、生成文本并打印模型信息、打印token数量、输出多条回答中的某一条)
'''

from Langchain.units import *     #从根目录开始导入
from langchain.llms import *

llm = OpenAI(model_name='text-ada-001', n=2,best_of=2)

# 基础功能-生成文本：输入一个str，返回一个str
print(llm("Tell a joke"))


# 基础功能-生成：输入列表，可以设置返回的数量，输出内容更加详细，会返回生成LLM提供者的详细信息
llm_result = llm.generate(["tell me a joke", "tell me a poem"]*15)
print(llm_result,'\n')                   # 返回所有的结果
print(llm_result.generations[0], '\n')   # 返回第0个结果
print(llm_result.generations[-1], '\n')  # 返回最后一个结果
print(llm_result.llm_output, '\n')       # 返回llm模型的信息
print(llm.get_num_tokens("what a joke")) # 可以计算输入的文本有多少个token