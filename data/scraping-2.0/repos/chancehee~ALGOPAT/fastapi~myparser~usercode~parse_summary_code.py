from langchain import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from myclass.usercode import UsercodeSummary
from logging import getLogger

# logger 설정 
logger = getLogger()

async def parse_summary_code(llm : LLMChain, text : str):
    parser = PydanticOutputParser(pydantic_object=UsercodeSummary)
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    formed_data = new_parser.parse(text)

    return formed_data
    
    
