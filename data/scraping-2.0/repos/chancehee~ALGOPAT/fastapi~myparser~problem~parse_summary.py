from langchain import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from myclass.problem import ProblemSummary
from logging import getLogger
from utils.log_decorator import log_decorator

# logger 설정 
logger = getLogger()

@log_decorator("json 타입으로 변환")
async def parse_summary(llm : LLMChain, text : str):
    parser = PydanticOutputParser(pydantic_object=ProblemSummary)
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    formed_data = new_parser.parse(text)

    return formed_data
    
    

