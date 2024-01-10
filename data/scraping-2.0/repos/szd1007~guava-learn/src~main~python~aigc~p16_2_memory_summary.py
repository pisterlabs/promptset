from langchain.memory import ConversationSummaryMemory
import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import ConversationChain
from  langchain.chains import SequentialChain
from langchain.chains import LLMRequestsChain
openai.api_key = os.environ.get("OPENAI_API_KEY")
import re
import json
from langchain.chains import TransformChain, SequentialChain


template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求：
1. 你的回答必须是中文
2. 回答限制在100个字内

{history}
Human: {input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)
llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=OpenAI())

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)
res_1 = conversation_with_summary.predict(input="你是谁")
res_2 = conversation_with_summary.predict(input="鱼香肉丝怎么做？")
res_3 = conversation_with_summary.predict(input="我往里面加辣椒可以么？")

print(res_1)
print(res_2)
print(res_3)

print(memory.load_memory_variables({}))