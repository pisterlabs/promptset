from injector import inject, singleton
from langchain_deepread.components.llm.llm_component import LLMComponent
from pydantic import BaseModel
import logging
from langchain_deepread.settings.settings import Settings
from langchain_deepread.server.moderation.moderation_model import OpenAIModerationChain

logger = logging.getLogger(__name__)


class GPTResponse(BaseModel):
    flag: bool
    msg: str


class CustomModeration(OpenAIModerationChain):
    def _moderate(self, text: str, results: dict) -> str:
        if results.flagged:
            error_str = f"The following text was found that violates DeepRead's content policy: {text}"
            if self.error:
                raise ValueError(error_str)
            else:
                return error_str
        return text


@singleton
class ModerationService:
    @inject
    def __init__(self, llm_component: LLMComponent, settings: Settings) -> None:
        self.llm_service = llm_component
        self.settings = settings

    def moderation(self, context: str) -> GPTResponse:
        custom_moderation = CustomModeration(
            error=True,
            openai_api_key=self.settings.openai.api_key,
            openai_api_base=self.settings.openai.api_base,
        )

        try:
            custom_moderation.run(context)
            return GPTResponse(flag=True, msg="")
        except ValueError as e:
            return GPTResponse(flag=False, msg=str(e))
