""" Using LLM to find service tpye from the question"""
from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain

from langchain.chains import QAWithSourcesChain

# from langchain.prompts.base import BasePromptTemplate

from langchain.schema import Document


PROMPT_TEMPLATE = """
    INSTRUCTIONS:
    Suppose you have some services which are described in documents. Please find the most relevant service to answer the question.
    If there is no relevant service, please indicate it in SOURCES field as [].
   
    请用中文回答
    
    QUESTION:
    {question}
"""

class ServiceTypeChain(Chain):
    """
    A Chain use QAWithSourcesChain to indentify service(s) to use.
    """
    llm: BaseLanguageModel
    answer_keys: List[str] = ["answer", "question", "sources", "additional_info"]
    question_key: str = "question"  # Input key for question
    baseChain: QAWithSourcesChain
    service_list: List[Dict]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return self.answer_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:


        # Create a LangChain prompt template that we can insert values to later
        prompt = PromptTemplate(
            input_variables=["question"],
            template=PROMPT_TEMPLATE,
        )

        list_doc = []
        for doc in self.service_list:
            text = doc["name"] + ":" + doc["description"]
            list_doc.append(Document(page_content=text, metadata={"source": doc["SID"]}))

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        try:
            if _run_manager:
                _run_manager.on_text("Checking for service type...")

            answer = self.baseChain(
                {
                    "question": prompt.format(question=inputs[self.question_key]),
                    "docs": list_doc
                },
                callbacks=_run_manager.get_child()
            )

            # 跟进原始信息
            answer["question"] = inputs[self.question_key]


            # 将结果的 Sources 转换为 List
            if isinstance(answer["sources"], str):
                answer["sources"] = answer["sources"].split(",")
                if answer["sources"][0] not in [services["SID"] for services in self.service_list]:
                    answer["sources"] = []
            else:
                answer["sources"] = []
                
            answer["additional_info"] = ""

            return answer

        except RuntimeError as exp:
            _run_manager.on_error(exp)

            return {
                "question": inputs[self.question_key],
                "answer": "",
                "sources": [],
                "additional_info": ""
            }


    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def _chain_type(self) -> str:
        return "ServiceTypeChain"

    @classmethod
    def from_list(
        cls,
        llm: BaseLanguageModel,
        service_list: List[Dict],
        **chain_type_kwargs,
    ):
        qa_chain = QAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type_kwargs=chain_type_kwargs,
        )

        # callbacks = chain_type_kwargs.get("callbacks", None)

        return cls(llm=llm, service_list=service_list, baseChain=qa_chain)


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

    test_llm = OpenAI(
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

    chain = ServiceTypeChain.from_list(
        llm=test_llm,
        service_list=services,
        verbose=True,
        callbacks=[StdOutCallbackHandler()]
    )

    res = chain({"question": "查询公司时候干什么的"})
    print(res)
