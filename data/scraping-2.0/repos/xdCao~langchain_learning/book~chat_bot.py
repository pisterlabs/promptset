from langchain.llms import OpenAI
import getpass
import os
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = getpass.getpass("输入apiKey: ")

llm = ChatOpenAI(temperature=0)

sys_prompt = SystemMessagePromptTemplate.from_template("你是一个很棒的翻译，可以将文字从{input_lang}翻译为{output_lang}")
human_prompt = HumanMessagePromptTemplate.from_template("{text}")

prompt = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])

chain = prompt | llm

response = chain.invoke({"input_lang": "中文", "output_lang": "英文", "text": "我想吃早餐"})
print(response)
