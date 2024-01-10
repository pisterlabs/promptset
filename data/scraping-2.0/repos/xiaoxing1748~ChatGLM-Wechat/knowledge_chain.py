# https://python.langchain.com/docs/integrations/llms/chatglm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
# from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import QianfanChatEndpoint
# from langchain_core.language_models.chat_models import HumanMessage
import os
import faiss_vector_store as faiss_vector_store
from langchain.chains import RetrievalQA
from llm import chatglm
import qianfan_api as qianfan

template = """基于以下【已知内容】，简短且专业地回答用户提出的【问题】，并遵循如下规则：
    1、【已知内容】中每个问答对以ask:开头，而回答以answer:开头。
    2、你的回答不应该以“根据已知内容”开头，请直接进行回答，不允许在答案中添加编造成分。

    【已知内容】:
    {context}
    
    【问题】:
    {question}
    
    【回答】:
    """


# Legacy LLM chain 这是示例，优先用qa_chain_legacy()
def llm_chain_legacy(question, docs):
    context = []
    # 遍历docs中的每个元素，提取page_content并添加到context
    for doc in docs:
        context.append(doc[0].page_content)

    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])
    model = chatglm()
    llm_chain = LLMChain(prompt=prompt, llm=model)
    return llm_chain.run(question=question, context=context)


# Legacy QA chain 专为QA设计的chain
def qa_chain_legacy(question, vector_store):
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])
    model = chatglm()
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain({"query": question})


# LCEL chain
def llm_chain(question):
    prompt = PromptTemplate.from_template("{question}")
    model = chatglm()
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser())
    return chain.invoke(question)


# LCEL QA chain
# https://python.langchain.com/docs/expression_language/how_to/map
def qa_chain(question, docs):
    context = []
    # 遍历docs中的每个元素，提取page_content并添加到context
    for doc in docs:
        context.append(doc[0].page_content)
    template = f"""基于以下【已知内容】，回答用户提出的【问题】，并遵循如下规则：
    1、每个问答对的问题以ask:开头，而回答以answer:开头。
    2、你的回答不应该以“根据已知内容”开头，请直接进行回答。
    3、如果无法从中得到答案，请说 "抱歉，我无法回答该问题"，此外不允许在答案中添加编造成分。

    【已知内容】:
    {context}"""+"""
    
    【问题】:
    {question}
    
    【回答】:
    """
    # retriever = vector_store.as_retriever()
    prompt = PromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    model = chatglm()
    chain = (
        {
            # retriever有点问题，会返回完整的元数据，改用列表暴力拼接
            # "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | output_parser
    )
    # return chain.invoke(question)
    return chain.invoke(question)


# qianfan chain
# https://python.langchain.com/docs/integrations/chat/baidu_qianfan_endpoint
def qianfan_chain(question, docs):

    # def qianfan_chain(accesskey, secretkey, content, model=None):
    # langchain框架的千帆模板坏了:qianfan.errors.AccessTokenExpiredError
    # os.environ["QIANFAN_AK"] = accesskey
    # os.environ["QIANFAN_SK"] = secretkey
    # chat = QianfanChatEndpoint(
    #     model=model,
    # )
    # response = chat([HumanMessage(content=content)])

    # 曲线救国一下:
    context = []
    # 遍历docs中的每个元素，提取page_content并添加到context
    for doc in docs:
        context.append(doc[0].page_content)
    content = f"""基于以下【已知内容】，简短且专业地回答用户提出的【问题】，并遵循如下规则：
    1、【已知内容】中每个问答对以ask:开头，而回答以answer:开头。
    2、你的回答不应该以“根据已知内容”开头，请直接进行回答，不允许在答案中添加编造成分。

    【已知内容】:
    {context}
    
    【问题】:
    {question}
    
    【回答】:
    """
    response = qianfan.chat(content)
    return response


if __name__ == '__main__':

    question = "丁真是理塘王吗？"
    # llm_run(question)
    docs = faiss_vector_store.search("question", "./document/news.txt")
    # vector_store = faiss_vector_store.index("./document/news1.txt")
    # print(qa_chain_legacy(question, vector_store))
    print(llm_chain(question, docs))
