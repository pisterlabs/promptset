# 操作输入输出
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain.callbacks import get_openai_callback

# 定义监听token函数
def track_token_usage(chain:RunnableSequence,query):
    with get_openai_callback() as callback:
        result = chain.invoke(query)
        print("token消费:",callback)
    print(result)

# 语料库
vectorstore = faiss.FAISS.from_texts(
    texts=["2023年亚运会在杭州举行","2022年冬奥会在北京举行","2021年世界杯在卡塔尔举行"], 
    embedding=OpenAIEmbeddings(
    )
)
retriever = vectorstore.as_retriever()


# 模板
template = """只回答关于Context里内容的问题。如果在Context内容未找到答案将如实告诉用户：抱歉我还未学习该知识，请谅解。:
Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# GPT
chat = ChatOpenAI(
  openai_api_key="",
  openai_api_base="",
  temperature=.7
)
# 模板结合
retriever_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)


track_token_usage(retriever_chain,"最近的冬奥会在哪举行？")
 




