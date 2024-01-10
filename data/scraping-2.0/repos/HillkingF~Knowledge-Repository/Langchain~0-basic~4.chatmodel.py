'''
@Author: 冯文霓
@Date: 2023/6/6
@Purpose: # 使用聊天模型完成聊天，每次可以发送一条或多条消息，响应将是一条消息
'''


from langchain.chat_models import ChatOpenAI
from Langchain.units import *     #从根目录开始导入

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


###################向聊天模型传入单条消息######################
# 1。获取聊天模型对象
chat = ChatOpenAI(temperature=0)
# 2. 向聊天模型中传入一条消息
x = chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# 3.获得响应。响应也是一条消息
print(x)
# 结果x： content="J'aime programmer." additional_kwargs={} example=False


###################向聊天模型传入多条消息######################
# 1。获取聊天模型对象
chat1 = ChatOpenAI(temperature=0)
# 2. 向聊天模型中传入一条消息
message = [
    SystemMessage(content='数仓建设标准是什么'),
    HumanMessage(content='翻译一下')
]
x = chat(message)
# 3.获得响应。响应也是一条消息
print(x)

# 结果x：content='What is the standard for building a data warehouse?' additional_kwargs={} example=False   —————— 可以看出响应对消息进行了翻译




###################聊天模型也可以与prompt、chain、代理、记忆一起使用######################
# 参考：https://python.langchain.com/en/latest/getting_started/getting_started.html



batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)