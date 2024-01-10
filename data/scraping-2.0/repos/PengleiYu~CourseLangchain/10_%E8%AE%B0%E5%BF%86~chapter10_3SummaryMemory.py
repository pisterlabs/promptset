from langchain.llms.openai import OpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryMemory

llm = OpenAI()
memory = ConversationSummaryMemory(llm=llm, verbose=True)
conv_chain = ConversationChain(llm=llm,
                               verbose=True,
                               memory=memory
                               )

result = conv_chain('我姐姐明天要过生日，我需要一束生日花束。')
print(result)
result = conv_chain('她喜欢粉色玫瑰，颜色是粉色的。"')
print(result)
result = conv_chain('我又来了，还记得我昨天为什么要来买花吗？')
print(result)
