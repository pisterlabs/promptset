"""
This is a module for the Michina test suite.
"""
from michina.checks.base.check import BaseCheck, BaseCheckResponse
from michina.exceptions.exceptions import (
    InvalidTypeException,
    InvalidXMLException,
    LanguageModelException,
)
from michina.checks.consistency.prompt import CONSISTENCY_CHECK_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from dotenv import load_dotenv
import xmltodict
from pydantic import BaseModel

load_dotenv()


class ConsistencyCheckInput(BaseModel):
    message: str
    statement: str


class ConsistencyCheckResponse(BaseCheckResponse):
    input: ConsistencyCheckInput
    reasoning: str
    judgment: float


class ConsistencyCheck(BaseCheck):
    description: str = "Checks whether the message is consistent with the statement."

    def check(self, message: str, statement: str) -> ConsistencyCheckResponse:
        prompt = PromptTemplate.from_template(CONSISTENCY_CHECK_TEMPLATE)
        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            string_response = chain.run(message=message, statement=statement)
        except Exception as e:
            raise LanguageModelException(e)

        def postprocessor(path, key, value):
            if key == "judgment":
                return key, float(value)
            else:
                return key, value

        try:
            dict_response = xmltodict.parse(
                string_response, postprocessor=postprocessor
            )
        except Exception as e:
            raise InvalidXMLException(e)

        try:
            response = ConsistencyCheckResponse(**dict_response["response"])
        except Exception as e:
            raise InvalidTypeException(e)

        return response
