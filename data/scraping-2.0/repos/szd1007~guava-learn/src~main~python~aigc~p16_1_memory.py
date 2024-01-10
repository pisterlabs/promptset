from langchain.memory import ConversationBufferWindowMemory
import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import LLMChain
from  langchain.chains import SequentialChain
from langchain.chains import LLMRequestsChain
openai.api_key = os.environ.get("OPENAI_API_KEY")
import re
import json
from langchain.chains import TransformChain, SequentialChain


template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求：
1. 你的回答必须是中文
2. 回答限制在100个字内

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
llm_chanin = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    memory=memory,
    verbose=True
)
res_1 = llm_chanin.predict(human_input="你是谁")
res_2 = llm_chanin.predict(human_input="鱼香肉丝怎么做？")
res_3 = llm_chanin.predict(human_input="我往里面加辣椒可以么？")

print(res_1)
print(res_2)
print(res_3)