'''
@Author: 冯文霓
@Date: 2023/6/6
@Purpose: 使用memory可以缓存上下文，记忆用户问题。 【多轮问答】
常用的方法是ConversationChai.ConversationChain有一种简单类型的内存，可以记住所有以前的输入/输出并将它们添加到传递的上下文中。
'''


from langchain import OpenAI, ConversationChain
from Langchain.units import *     #从根目录开始导入
llm = OpenAI(temperature=0)

# ConversationChain有一种简单类型的内存，可以记住所有以前的输入/输出并将它们添加到传递的上下文中。
conversation = ConversationChain(llm=llm, verbose=True)


i = 0
while i < 10:
    str = input("请输入：")
    output = conversation.predict(input=str)
    print(output)
    i = i + 1



"""  回答示例：多轮问答
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: a = 1, b = 1, c = a+b,c等于几
AI:  c等于2
Human: d = c +a,d等于几
AI:  d等于3
Human: e=d+c,e?
AI:

> Finished chain.
 e等于5：

"""