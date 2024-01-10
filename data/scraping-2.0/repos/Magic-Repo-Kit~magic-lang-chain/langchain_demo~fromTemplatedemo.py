from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import get_openai_callback

prompt = ChatPromptTemplate.from_template("我想去 <{topic}> 旅行，我想知道这个地方有什么好玩的")

# 提示词模板
chat = ChatOpenAI(
    openai_api_key="",
    openai_api_base="",
    temperature=.7
                 )

# 输出模板
print("输出模板:",prompt.messages[0].prompt)

# token消费回调
with get_openai_callback() as callback:
    # 输出解析器
    output_parser = StrOutputParser()

    chain = prompt | chat | output_parser

    print("AI:",chain.invoke({"topic": "广州"}))

    print("token消费:",callback)

    



    