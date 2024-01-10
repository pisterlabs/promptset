"""
大语言模型是无状态的，它并不保存上次交互的内容，chatGPT能够和人正常对话，因为它做了一层包装，把历史记录传回给了模型。
为了解决这个问题，LangChain提供了记忆组件。记忆有两种类型：短期和长期记忆。短期记忆一般指单一会话时传递数据，长期记忆则是处理多个会话时获取和更新信息。
目前记忆组件只需要考虑ChatMessageHistory ，我们看个例子：
"""
from langchain.memory import ChatMessageHistory
from langchain import ConversationChain
from langchain.llms import OpenAI
import os
from langchain.schema import messages_from_dict, messages_to_dict
os.environ["OPENAI_API_KEY"] = "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"


history = ChatMessageHistory()
history.add_user_message("你好")
history.add_ai_message("有什么事情吗？")
print(history.messages)
print("--------------------------------------------------")
llm = OpenAI(temperature=0.9)
conversaation = ConversationChain(llm=llm,verbose=True)
str1 = conversaation.predict(input="小明有1只猫")
str2 = conversaation.predict(input="小明有2只狗")
str3 = conversaation.predict(input="小明和小刚一共有几只宠物？")
print(str1)
# print(str2)
# print(str3)
print("--------------------------------------------------")
dicts = messages_to_dict(history.messages)
print(dicts)
print("--------------------------------------------------")
new_messages = messages_from_dict(dicts)
print(new_messages)


