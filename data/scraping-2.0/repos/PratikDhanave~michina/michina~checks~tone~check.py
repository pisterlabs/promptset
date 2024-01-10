from michina.checks.base.check import BaseCheck, BaseCheckResponse
from michina.checks.tone.prompt import TONE_CHECK_TEMPLATE
from michina.exceptions.exceptions import (
    InvalidTypeException,
    InvalidXMLException,
    LanguageModelException,
)
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from dotenv import load_dotenv
import xmltodict

load_dotenv()


class ToneCheckInput(BaseModel):
    message: str
    tone: str


class ToneCheckResponse(BaseCheckResponse):
    input: ToneCheckInput
    reasoning: str
    judgment: float


class ToneCheck(BaseCheck):
    description: str = (
        "Checks whether the tone of a given message matches the tone provided."
    )

    def check(self, message: str, tone: str) -> ToneCheckResponse:
        prompt = PromptTemplate.from_template(TONE_CHECK_TEMPLATE)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            string_response = chain.run(message=message, tone=tone)
        except Exception as e:
            raise LanguageModelException(e)

        try:
            response = xmltodict.parse(string_response)
        except Exception as e:
            raise InvalidXMLException(e)

        try:
            response = ToneCheckResponse(**response["response"])
        except Exception as e:
            raise InvalidTypeException(e)

        return response
