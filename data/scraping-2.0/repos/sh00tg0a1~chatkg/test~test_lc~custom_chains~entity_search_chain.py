""" Search for the entities mentioned in the question"""
from typing import Any, Dict, List, Optional
import json
from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain

from langchain.chains import LLMChain

import uuid


PROMPT_TEMPLATE = """
    INSTRUCTIONS:
    You are helpful of information extraction. You can extract entities from any sentence. And you can do it in Chinese.
    Please recognize the entities or properties mentioned the in question. You are trying to extract {entity_number} entities.
    
    Entity_types:
    {entity_types}
    
    EXAMPLES:
    question:
    {{
        "question": "上海爱数和长沙爱数的关系是什么?",
        "entity_number": 2,
        "entity_types": ["Enterprise"],
    }}
    answer:
    {{
        "entities": [
            {{
                "type": "Enterprise",
                "name": "上海爱数"
            }},
            {{
                "type": "Enterprise",
                "name": "长沙爱数"
            }}
       ],
       "message": "I extracted 2 entities. Which are: 上海爱数, 长沙爱数."
    }}
    
    If you can't extract all entities, please just extract as many as you can.
    
    DO NOT try to translate question into English. And anwser should be in Chinese.
    
    QUESTION:
    {question}

    ANSWER:
"""

# 需要替换成真实的查询 vid 的接口
def get_vid(sid: str, name: str) -> str:
    """get vid from KG according to sid and vid

    Args:
        sid (str): service id
        name (str): entity name

    Returns:
        str: vid of the entity
    """
    return "vid-" + uuid.uuid4().hex


class EntitySearchChain(Chain):
    """
    Using LLM to extract entities from question and search entity ids from KG.
    """
    llm: BaseLanguageModel
    answer_keys: List[str] = ["question", "entities", "addition_info"]
    question_key: str = "question"  # Input key for question
    baseChain: LLMChain

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

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        try:
            if _run_manager:
                _run_manager.on_text("Extracting entities from question...")

            res = self.baseChain(
                {
                    "question": inputs[self.question_key],
                    "entity_types": inputs.get("entity_types", [""]),
                    "entity_number" : inputs.get("entity_number", 1)
                },
                callbacks=_run_manager.get_child()
            )

            res["answer"] = json.loads(res["answer"])

            for entity in res["answer"]["entities"]:
                entity["vid"] = get_vid(inputs.get("sid", "1"), entity["name"])

            return {
                "question": inputs[self.question_key],
                "entities": res["answer"]["entities"],
                "addition_info": res["answer"]["message"]
            }

        except RuntimeError as exp:
            if _run_manager:
                _run_manager.on_chain_error(exp)

            return {
                "question": inputs[self.question_key],
                "entities": [""],
                "addition_info": ""
            }


    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def _chain_type(self) -> str:
        return "EntitySearchChain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        **kwargs,
    ):
        """ create a EntitySearchChain from a language model."""

        # callbacks = chain_type_kwargs.get("callbacks", None)
        entity_prompt = PromptTemplate(
            input_variables=["entity_number", "question", "entity_types"],
            template=PROMPT_TEMPLATE,
        )

        extract_chain = LLMChain(llm=llm, prompt=entity_prompt, output_key="answer", **kwargs)
        return cls(llm=llm, baseChain=extract_chain)


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


    # services = [
    #     {
    #         "SID": "1",
    #         "name": "企业查询",
    #         "description": "查询企业的基本信息，如成立时间、注册资本、注册地址、经营范围等",
    #         "params": [
    #             {
    #                 "name": "vid",
    #                 "type": "vid",
    #                 "entity_type": "Enterprise",
    #                 "description": "知识图谱中的企业ID",
    #             }
    #         ]
    #     },
    #     {
    #         "SID": "2",
    #         "name": "企业间投资路径查询",
    #         "description": "查询两个企业间的投资路径, 如A企业投资B企业, B企业投资C企业, 那么A企业和C企业间就存在投资路径",
    #         "params": [
    #             {
    #                 "name": "from_vid",
    #                 "type": "vid",
    #                 "entity_type": "Enterprise",
    #                 "description": "知识图谱中的企业ID",
    #             },
    #             {
    #                 "name": "to_vid",
    #                 "type": "vid",
    #                 "entity_type": "Enterprise",
    #                 "description": "知识图谱中的企业ID",
    #             },
    #         ]
    #     },
    #     {
    #         "SID": "3",
    #         "name": "企业实控人分析",
    #         "description": "查询企业的实际控制人，以及实际控制人的股权结构",
    #         "params": [
    #             {
    #                 "name": "vid",
    #                 "type": "vid",
    #                 "entity_type": "Enterprise",
    #                 "description": "知识图谱中的企业ID",
    #             }
    #         ]
    #     }
    # ]

    chain = EntitySearchChain.from_llm(
        llm=test_llm,
        verbose=True,
        # callbacks=[StdOutCallbackHandler()]
    )

    test_res = chain({"question": "上海爱数成立于哪一年", "entity_types": ["Enterprise"], "entity_number": 2, "sid": "1"})
    print(test_res)
