'''
@Author: 冯文霓
@Date: 2023/6/6
@Purpose: langchain有两种不同的子模型，LLMs和ChatModels。其中llms传输文本，聊天模型传输消息
下面看一下这两个子模型的区别
'''

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from Langchain.units import *     #从根目录开始导入


llm = OpenAI()
chat = ChatOpenAI()

print(llm.predict("你好"))
print(chat.predict("你好"))


""" 两个子模型的回答结果如下：存在差异
你好！很高兴见到你！
你好，有什么可以帮助您的吗？
"""



