from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.schema.output_parser import BaseOutputParser


def get_retry_parser(parser: BaseOutputParser):
    return RetryWithErrorOutputParser.from_llm(parser=parser, llm=ChatOpenAI(temperature=0))
