# 对话模板
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate


def makeChain(retriever):
    
    from internal.model import glmModel
    model = glmModel()
    
    template = """
    你是某软件项目的测试人员，需要根据需求规格说明文档的内容编写测试用例。接下来我会给出相关的需求规格文档，而你将根据其写出对应的测试用例。
    要求测试用例格式为用严格合法json的形式，并要求包括案例序号，案例名称，描述，参与者，触发条件，前置条件，后置条件，正常流程，预期值。其中正常流程尽可能写全面点。
    下面是相关的需求规格文档：
    {context}

    要求写出 {question}的测试用例
    """
    prompt = ChatPromptTemplate.from_template(template)

    if retriever is None:
        chain = (
            {"context": lambda *args, **kwargs: "", "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return chain
    
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain


def _format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])