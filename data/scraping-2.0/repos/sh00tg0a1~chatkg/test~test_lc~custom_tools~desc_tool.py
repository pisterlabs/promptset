from typing import Optional, List, Dict, Type
import json

from pydantic import (
    Field,
    BaseModel
)

from langchain.tools import (
    BaseTool,
    # StructuredTool,
    # Tool,
    # tool
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.chains import RetrievalQAWithSourcesChain, QAWithSourcesChain
from langchain.schema import Document, BaseRetriever
from langchain.base_language import BaseLanguageModel

from langchain import PromptTemplate


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



PROMPT_TEMPLATE = """
    %INSTRUCTIONS:
    Suppose you have some services which are described in documents. Please find the most relevant service to answer the question.
    If more than one service is relevant, please indicate them in [sources] list.
    
    Please also put the {question} in the [query] field.
    
    请用中文回答
    
    %QUESTION:
    {question}
"""


class DescToolInput(BaseModel):
    """ input for desc tool"""
    question: str = Field()


class DescTool(BaseTool):
    """ tool to get usage from desc docs"""

    name = "desc_retriever"
    services_desc: List[Dict] = None
    description = "suppose you can provide some services. When users ask you what you can do, and you can find the answer as service listed: {services_desc}"
    args_schema: Type[BaseModel] = DescToolInput

    llm: BaseLanguageModel = None
    """Example:
        [{
            "SID": "1",
            "name": "企业查询",
            "description": "查询企业的基本信息，如成立时间、注册资本、注册地址、经营范围等",
        }],
    """

    @DeprecationWarning
    def _qa(self, question, verbose=False):
        """_summary_
        """
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            retriever=ListRetriever.from_list(self.services_desc, key="SID"),
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

    # 更加简洁的视线，不需要使用 Retriever，而是直接使用 QAWithSourcesChain
    def _qa_2(self, question, docs, verbose=False):
        qa_chain = QAWithSourcesChain.from_chain_type(
            llm=self.llm,
            verbose=verbose)

        # Create a LangChain prompt template that we can insert values to later
        prompt = PromptTemplate(
            input_variables=["question"],
            template=PROMPT_TEMPLATE,
        )
        
        list_doc = []
        for doc in docs:
            text = doc["name"] + ":" + doc["description"]
            list_doc.append(Document(page_content=text, metadata={"source": doc["SID"]}))

        # docs 里面的内容是需要查询的来源
        answer = qa_chain({
            "question": prompt.format(question=question),
            "docs": list_doc
        })
        return answer

    def _run(
        self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # return self._qa(question)
        
        return self._qa_2(question, self.services_desc)

    async def _arun(
        self, question: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DescTool does not support async")

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, services_desc: list):
        # cls.description.format(desc_docs=json.dumps(desc_docs))
        tool = cls(llm=llm, services_desc=services_desc)
        tool.description = tool.description.format(
            services_desc="\n".join([desc["name"] + ": " + desc["description"] for desc in services_desc])
        )

        return tool


if __name__ == "__main__":
    from langchain import OpenAI
    import openai
    import os
    
    import dotenv

    env_file = '../../../.env'
    dotenv.load_dotenv(env_file, override=True)


    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_ENDPOINT  = os.environ.get("OPENAI_ENDPOINT")
    OPENAI_API_VERSION  = os.environ.get("OPENAI_API_VERSION")
    OPENAI_API_TYPE  = os.environ.get("OPENAI_API_TYPE")

    openai.api_type = OPENAI_API_TYPE
    openai.api_base = OPENAI_ENDPOINT
    openai.api_version = OPENAI_API_VERSION
    openai.api_key = OPENAI_API_KEY

    llm = OpenAI(
        # engine = "asada001",
        # engine="aschatgpt35",
        engine="asdavinci003",
        # model_kwargs= {
        #     "engine": "asdavinci003",
        # },
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_ENDPOINT,
    )

    services = [
        {
            "SID": "1",
            "name": "企业查询",
            "description": "查询企业的基本信息，如成立时间、注册资本、注册地址、经营范围等",
            "params": [
                {
                    "name": "vid",
                    "type": "vid",
                    "entity_type": "Enterprise",
                    "description": "知识图谱中的企业ID",
                }
            ]
        },
        {
            "SID": "2",
            "name": "企业间投资路径查询",
            "description": "查询两个企业间的投资路径, 如A企业投资B企业, B企业投资C企业, 那么A企业和C企业间就存在投资路径",
            "params": [
                {
                    "name": "from_vid",
                    "type": "vid",
                    "entity_type": "Enterprise",
                    "description": "知识图谱中的企业ID",
                },
                {
                    "name": "to_vid",
                    "type": "vid",
                    "entity_type": "Enterprise",
                    "description": "知识图谱中的企业ID",
                },
            ]
        },
        {
            "SID": "3",
            "name": "企业实控人分析",
            "description": "查询企业的实际控制人，以及实际控制人的股权结构",
            "params": [
                {
                    "name": "vid",
                    "type": "vid",
                    "entity_type": "Enterprise",
                    "description": "知识图谱中的企业ID",
                }
            ]
        }
    ]

    tool = DescTool.from_llm(llm=llm, services_desc=services)
    print(tool.run("你能查实控人吗?请给我 Json 的回答，不需要原始的问题。如果你需要更多信息可以告诉我"))
