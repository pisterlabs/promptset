from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Type
import json

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

from langchain.chains import LLMChain
from langchain.base_language import BaseLanguageModel

from langchain import PromptTemplate




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
                "enterprise": "上海爱数"
            }},
            {{
                "enterprise": "长沙爱数"
            }}
       ]
    }}
    
    If you can't extract all entities, please just extract as many as you can.
    
    DO NOT try to translate question into English.
    
    QUESTION:
    {question}

    ANSWER:
"""

entity_prompt = PromptTemplate(
    input_variables=["entity_number", "question", "entity_types"],
    template=PROMPT_TEMPLATE,
)


class EntitySearchInput(BaseModel):
    question: str = Field(..., description="question to extract entities from")
    entity_types: List[str] = Field(["Enterprise"], description="entity types to extract")
    entity_number: int = Field(1, ge=1, description="number of entities to extract")


class EntitySearchTool(BaseTool):
    """ tool to get usage from desc docs
    """
    name = "EntitySearch"
    description = "useful to extract entities from question following indicated entity or property types with disignated number."
    args_schema: Type[BaseModel] = EntitySearchInput

    llm: BaseLanguageModel = None

    def _run(
        self,
        question: str,
        entity_types: List[str],
        entity_number: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        extract_chain = LLMChain(llm=self.llm, prompt=entity_prompt)

        res = extract_chain.run({
            "question": question,
            "entity_types": entity_types,
            "entity_number" : entity_number
        })

        return {
            "question": question,
            "answer": json.loads(res),
        }

    async def _arun(
        self, question: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("EntitySearch does not support async")

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel):
        return cls(llm=llm)


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

    tool = EntitySearchTool.from_llm(llm=llm)
    tool.run({"question": "上海爱数的实控人是谁", "entity_types": ["Enterprise"], "number": 1})
