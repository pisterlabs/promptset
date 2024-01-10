'''
@Author: 冯文霓
@Date: 2023/6/7
@Purpose: 如何在磁盘上写入和读取 LLM 配置.LLM 可以以两种格式保存在磁盘上：json 或 yaml
'''

from Langchain.units import *
from langchain.llms import OpenAI
from langchain.llms.loading import load_llm
from langchain.llms import OpenAI

# 从磁盘上加载LLm
llm_path = 'llm.json'   # 或llm.yaml文件
llm = load_llm(llm_path)
print(llm)


# 将模型保存到磁盘上
llm1 = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
llm1.save('llm1.json')
