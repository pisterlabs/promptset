
import os
os.environ['OPENAI_API_KEY'] = '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'

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

#在此处进行prompt的修改，角色任务的设定
template = """Json reads some assets and the proportion to be allocated. Please imitate the professional tone of the fund manager and help me explain the results of asset allocation for my clients."""

#读取json后，将json文件转换为文本格式，方便输入
import json

#此处需要重新操作一下，如果发现文件更新，需要重新阅读一遍，或者有没有更加显示的操作
#比如matlab端口
with open('C:\\Users\\11834\\Desktop\\results.json') as f:
    data = json.load(f)

text = json.dumps(data)
print(text)

from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(openai_api_key='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')

from langchain.schema.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content=template),
    HumanMessage(content=text),
]

Asset_Result = chat.invoke(messages)

print(chat.invoke(messages))