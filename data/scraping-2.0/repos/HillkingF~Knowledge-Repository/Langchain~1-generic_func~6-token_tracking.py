'''
@Author: 冯文霓
@Date: 2023/6/7
@Purpose: 如何跟踪特定呼叫的令牌使用情况。它目前仅针对 OpenAI API 实现。
'''

from Langchain.units import *

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)