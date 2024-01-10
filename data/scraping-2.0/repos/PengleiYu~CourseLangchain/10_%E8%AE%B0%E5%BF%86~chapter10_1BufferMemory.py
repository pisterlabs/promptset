from langchain.llms.openai import OpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI()
memory = ConversationBufferMemory()
conv_chain = ConversationChain(llm=llm,
                               verbose=True,
                               memory=memory
                               )
# print(conv_chain.prompt)

conv_chain.run('我姐姐明天要过生日，我需要一束生日花束。')
print('第一轮对话的记忆:', memory.buffer)
conv_chain.run('她喜欢粉色玫瑰，颜色是粉色的。"')
print('第二轮对话的记忆:', memory.buffer)
conv_chain('我又来了，还记得我昨天为什么要来买花吗？')
print('第三轮时的提示:', conv_chain.prompt)
print('第三轮对话的记忆:', memory.buffer)
