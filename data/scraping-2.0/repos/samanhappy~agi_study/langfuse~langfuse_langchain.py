from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langfuse.callback import CallbackHandler
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

handler = CallbackHandler(
    os.getenv("LANGFUSE_PUBLIC_KEY"),
    os.getenv("LANGFUSE_SECRET_KEY"),
    os.getenv("LANGFUSE_HOST"),
)

# 定义语言模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

# 定义Prompt模板
prompt = PromptTemplate.from_template("Say hello to {input}!")

# 定义输出解析器
parser = StrOutputParser()

chain = {"input": RunnablePassthrough()} | prompt | llm | parser

chain.invoke("中国", config={"callbacks": [handler]})
