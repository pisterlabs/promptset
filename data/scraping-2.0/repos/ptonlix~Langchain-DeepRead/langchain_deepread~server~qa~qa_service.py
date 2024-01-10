from injector import inject, singleton
from langchain_deepread.components.llm.llm_component import LLMComponent
from langchain.schema.output_parser import StrOutputParser
from langchain_deepread.server.qa.qa_prompt import (
    QAParam,
    QAPromptTemplate,
)
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
import json
import logging
from langchain_deepread.settings.settings import Settings

logger = logging.getLogger(__name__)


class GPTResponse(BaseModel):
    answer: str


class TokenInfo(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0


class QAResponse(BaseModel):
    gptresponse: GPTResponse = Field(description="gpt response qa")
    token_info: TokenInfo = Field(description="Token Info")


@singleton
class QAService:
    @inject
    def __init__(self, llm_component: LLMComponent, settings: Settings) -> None:
        self.settings = settings
        if self.settings.llm.mode == "mock":
            llm_component.llm = "qa"
        self.llm_service = llm_component

    def qa(self, question: str, context: str) -> QAResponse | None:
        resobj = None
        try:
            sp = QAParam(question=question, context=context)
            output_parser = StrOutputParser()
            chain = QAPromptTemplate | self.llm_service.llm | output_parser
            with get_openai_callback() as cb:
                ret = chain.invoke(sp.model_dump())
                gptresponse = json.loads(ret)
                tokeninfo = TokenInfo(
                    total_tokens=cb.total_tokens,
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens,
                    successful_requests=cb.successful_requests,
                    total_cost=cb.total_cost,
                )
                resobj = QAResponse(
                    token_info=tokeninfo, gptresponse=GPTResponse(**gptresponse)
                )

        except Exception as e:
            logger.exception(e)
        return resobj
