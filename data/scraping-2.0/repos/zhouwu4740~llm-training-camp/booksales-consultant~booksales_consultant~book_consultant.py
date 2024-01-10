from operator import itemgetter
from typing import Literal

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.vectorstores.faiss import FAISS
from pydantic.v1 import BaseModel
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

from prompt import prompt
from utils import logger
from shopping_cart import finish_request


def determine_topic(question: str) -> Literal["qa", "shopping"]:
    """Determine the topic of the question
    qa: Reply to user inquiries for books and provide some purchasing suggestions
    shopping: Assist customers in managing their shopping cart, including services such as adding products,
        deleting products, viewing inventory, and settling accounts
    """
    classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
    llm = ChatOpenAI(temperature=0).bind(functions=[classifier_function], function_call={"name": "TopicClassifier"})
    parser = PydanticAttrOutputFunctionsParser(pydantic_schema=TopicClassifier, attr_name="topic")
    classifier_chain = llm | parser

    return classifier_chain.invoke(question)


def question_answer() -> Runnable:
    embedding = OpenAIEmbeddings()

    db = FAISS.load_local('../jupyter/books', embedding)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7, "k": 3})
    llm = ChatOpenAI(temperature=0.7)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True)
    qa.combine_documents_chain.verbose = True
    qa.return_source_documents = True
    # TODO 根据配置选择是否需要修改 prompt 模板
    qa.combine_documents_chain.llm_chain.prompt = prompt

    return qa


def final_chain(qa: Runnable = None):
    # branches = RunnableBranch(
    #     (lambda p: p['topic'] == "qa", qa),
    #     (lambda p: p['topic'] == "shopping",
    #      RunnablePassthrough.assign(result=lambda x: '购物车功能还在抓紧开发中，敬请期待！')),
    #     lambda x: x
    # )
    branches = RunnableBranch(
        (lambda p: p['topic'] == "qa", qa),
        (lambda p: p['topic'] == "shopping",
         RunnablePassthrough.assign(request=itemgetter('query')) | RunnableLambda(finish_request)),
        lambda x: x
    )

    chain = (
            RunnablePassthrough.assign(topic=itemgetter('query') | RunnableLambda(determine_topic))
            | branches
    )
    return chain.with_fallbacks(
        fallbacks=[RunnableLambda(default_function)])
    # return chain


def default_function(x):
    logger.error(f"正常流程调用出错. {x}")
    return {"result": "您的请求有点小小难度，我需要升级大脑后才能给你服务:-("}


class BookConsultant:
    """Book Consultant
        1. Reply customer's any question about the book
        2. Help customer to shopping.eg, add book to shopping cart,
            delete book from shopping cart, view inventory, checkout
    """
    history_size: int

    def __init__(self, history_size: int = 10) -> None:
        self.history_size = history_size
        self.qa = question_answer()
        self.consultant = final_chain(self.qa)
        # 历史对话记录
        self.history = []

    def ask(self, question: str):
        logger.info(f"question: {question}")

        reply = self.consultant.invoke({"query": question,
                                        "history": ChatPromptTemplate.from_messages(self.history)})
        logger.info(f"reply: {reply}")
        reply_content = reply["result"]
        # 添加历史记录
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=reply_content))
        # 保留最近的历史记录
        self.history = self.history[-self.history_size:]
        return reply_content


class TopicClassifier(BaseModel):
    """Classify the topic of the user question
    qa: Reply to user inquiries for books and provide some purchasing suggestions
    shopping: Assist customers in managing their shopping cart, including services such as adding products,
        deleting products, viewing inventory, and settling accounts
    """

    topic: Literal["qa", "shopping"]
    """The topic of the user question. One of 'qa' or 'shopping'."""


CONSULTANT = BookConsultant()

if __name__ == '__main__':
    consultant = BookConsultant()
    # reply_result = consultant.ask("你这里有 Python 的书吗？")
    # logger.info(f"reply result: {reply_result}")

    reply_result = consultant.ask("帮我加入购物车")
    logger.info(f"reply result: {reply_result}")
