from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

set_debug(True)

cot_prompt = PromptTemplate.from_template(
    """以下の質問に回答してください。

{question}
                                       
ステップバイステップで考えてください。"""
)

summarize_prompt = PromptTemplate.from_template(
    """結論だけ要約してください。

{text}"""
)
model = ChatOpenAI(model="gpt-3.5-turbo-1106")

chain1 = {"question": RunnablePassthrough()} | cot_prompt | model | StrOutputParser()
chain2 = {"text": RunnablePassthrough()} | summarize_prompt | model | StrOutputParser()

chain = chain1 | chain2

result = chain.invoke("3*3+4*4")
print(result)
