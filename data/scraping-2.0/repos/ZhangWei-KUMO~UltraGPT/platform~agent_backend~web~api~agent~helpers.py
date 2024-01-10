from typing import TypeVar

from langchain import LLMChain, BasePromptTemplate
from langchain.prompts import PromptTemplate
 
from langchain.schema import OutputParserException, BaseOutputParser
from openai import InvalidRequestError
from openai.error import ServiceUnavailableError
from langchain.llms import OpenAI

from agent_backend.schemas import ModelSettings
from agent_backend.web.api.agent.model_settings import create_model
from agent_backend.web.api.errors import OpenAIError

T = TypeVar("T")

def parse_with_handling(parser: BaseOutputParser[T], completion: str) -> T:
    try:
        return parser.parse(completion)
    except OutputParserException as e:
        raise OpenAIError(
            e, "解析 AI 模型响应时出现问题。"
        )

# 调用模型进行处理
async def call_model_with_handling(
    model_settings: ModelSettings,
    prompt: BasePromptTemplate, 
    args: dict[str, str]
) -> str:
    try:
        llm = create_model(model_settings)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(args)
        return result
    except ServiceUnavailableError as e:
        raise OpenAIError(
            e,
            "OpenAI 出现问题，请访问 "
            "https://status.openai.com/",
        )
    except InvalidRequestError as e:
        raise OpenAIError(e, e.user_message)
    except Exception as e:
        raise OpenAIError(e, "从 AI 模型获取响应时出现问题。")
