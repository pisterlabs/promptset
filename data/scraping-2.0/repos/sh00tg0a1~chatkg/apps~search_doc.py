from typing import Dict, List
import os

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document, BaseRetriever
# from langchain.embeddings.openai import OpenAIEmbeddings

import openai


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ENDPOINT  = os.environ.get("OPENAI_ENDPOINT")
OPENAI_API_VERSION  = os.environ.get("OPENAI_API_VERSION")
OPENAI_API_TYPE  = os.environ.get("OPENAI_API_TYPE")

openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_ENDPOINT
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY

LLM = OpenAI(
    # engine = "asada001",
    # engine="aschatgpt35",
    engine="asdavinci003",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_ENDPOINT,
    max_tokens=256,
)

# LLM = ChatOpenAI(
#     # engine = "asada001",
#     engine="aschatgpt35",
#     # engine="asdavinci003",
#     openai_api_key=OPENAI_API_KEY,
#     openai_api_base=OPENAI_ENDPOINT,
#     max_tokens=256,
# )

services = [
    {
        "SID": "1",
        "name": "企业查询",
        "description": "查询企业的基本信息，如成立时间、注册资本、注册地址、经营范围等",
    },
    {
        "SID": "2",
        "name": "企业间投资路径查询",
        "description": "查询两个企业间的投资路径, 如A企业投资B企业, B企业投资C企业, 那么A企业和C企业间就存在投资路径",
    },
    {
        "SID": "3",
        "name": "企业实控人分析",
        "description": "查询企业的实际控制人，以及实际控制人的股权结构",
    },
    {
        "SID": "4",
        "name": "企业舆情图谱",
        "description": "查询企业的舆情信息，包含情报、新闻，以及正面、负面的情况",
    },
    {
        "SID": "5",
        "name": "产业链查询",
        "description": "查询企业的产业链，包含上游、下游企业，企业的产品、服务等",
    },
    {
        "SID": "6",
        "name": "企业股东查询",
        "description": "查询企业的股东信息，包含股东的股权结构、股东的投资信息等",
    },
    {
        "SID": "7",
        "name": "企业关系查询",
        "description": "查询企业的之间的关系, 企业A和企业B之间的关系, 企业A和企业B之间的路径",
    },
]


__STORE_PATH = os.path.join(os.path.dirname(__file__), "store")
__STORE_INSTANCE = None
__EMBEDDING_INSTANCE = None

def get_faiss(force_reindex=False):
    """_summary_
    
    args:
        force_reindex (bool): 是否强制重新建立索引
    """

    global __EMBEDDING_INSTANCE
    if __EMBEDDING_INSTANCE is None:
        # __EMBEDDING_INSTANCE = OpenAIEmbeddings(
        #     deployment="astextada002",
        #     openai_api_key=OPENAI_API_KEY,
        #     openai_api_base=OPENAI_ENDPOINT,
        #     openai_api_type=OPENAI_API_TYPE,
        #     openai_api_version=OPENAI_API_VERSION,
        # )
        
        model_name = "moka-ai/m3e-base"
        model_kwargs = {'device': 'cpu'}
        __EMBEDDING_INSTANCE = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
        )

    global __STORE_INSTANCE
    if __STORE_INSTANCE is not None:
        return __STORE_INSTANCE
    try:
        # 强制重新加载索引
        if force_reindex:
            raise RuntimeError
        
        __STORE_INSTANCE = FAISS.load_local(
            folder_path=__STORE_PATH,
            embeddings=__EMBEDDING_INSTANCE,
            index_name="services")

    except RuntimeError:
        __STORE_INSTANCE = FAISS.from_texts(
            texts=[service["name"] + ":" + service["description"] for service in services],
            metadatas=[{"source":service["SID"]} for service in services],
            embedding=__EMBEDDING_INSTANCE
        )

        __STORE_INSTANCE.save_local(folder_path=__STORE_PATH, index_name="services")
    return __STORE_INSTANCE


def get_service(question):
    """_summary_

    Args:
        question (_type_): _description_

    Returns:
        _type_: _description_
    """
    return get_faiss().similarity_search_with_score(question, k=3)

PROMPT_TEMPLATE = """
    %INSTRUCTIONS:
    Suppose you have some services which are described in documents. Please find the most relevant service to answer the question.
    If more than one service is relevant, please indicate them in [sources] list.
    
    Please also put the {question} in the [query] field.
    
    请用中文回答
    
    %QUESTION:
    {question}
"""

def qa_with_index(question, index, verbose=False) -> str:
    """_summary_

    """
    if index is None:
        index = get_faiss()

    # 根据调试的结果，这里从 index 中默认会过去 4 个结果，而且无法修改
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=LLM,
        retriever=index.as_retriever(),
        verbose=verbose)

    # Create a LangChain prompt template that we can insert values to later
    prompt = PromptTemplate(
        input_variables=["question"],
        template=PROMPT_TEMPLATE,
    )

    answer = qa_chain({"question": prompt.format(question=question)})
    return answer


def qa_without_index(question, verbose=False):
    """_summary_
    """
    class ListRetriever(BaseRetriever):
        """将list转换为retriever

        Args:
            BaseRetriever (_type_): _description_
        """
        def __init__(self, list_doc: List[Document]) -> None:
            self.list_doc = list_doc

        @classmethod
        def from_list(cls, input_list: List[Dict], key: str = "source"):
            """将list转换为retriever

            Args:
                input_list (List[Dict]): _description_
                key (str, optional): _description_. Defaults to "source".

            Returns:
                _type_: _description_
            """
            list_doc = []
            for doc in input_list:
                text = doc["name"] + ":" + doc["description"]
                list_doc.append(Document(page_content=text, metadata={"source": doc[key]}))

            return cls(list_doc)

        def get_relevant_documents(self, query: str) -> List[Document]:
            return  self.list_doc

        async def aget_relevant_documents(self, query: str) -> List[Document]:
            return  self.list_doc

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=LLM,
        retriever=ListRetriever.from_list(services, key="SID"),
        verbose=verbose)

    # Create a LangChain prompt template that we can insert values to later
    prompt = PromptTemplate(
        input_variables=["question"],
        template=PROMPT_TEMPLATE,
    )

    answer = qa_chain({
        "question": prompt.format(question=question),
    })
    return answer


if __name__ == "__main__":
    # print(get_service("长沙爱得自在的成立年份？"))
    # print(get_service("上海爱数和长沙爱得自在的关系？"))

    # service_index = get_faiss()
    # res = qa_with_index("爱数近期有哪些新闻?")
    # print(res)

    res = qa_without_index("爱数近期有哪些新闻?")
    print(res)

    # res = qa_without_index("上海爱数和长沙爱得自在的关系？")
    # print(res)

    # res = qa_without_index("长沙爱得自在的最终控制人？")
    # print(res)

    # service_index = get_faiss()
    # res = qa_with_index("长沙爱得自在的最终控制人？", service_index)
    # print(res)
