# 对于格式不正确的输出如何修复

from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field


class Flower(BaseModel):
    name: str = Field(description='name of flower')
    colors: list[str] = Field(description='the colors of flower')


flower_query = "Generate the characters for a random flower."

mis_formatted = "{'name': '康乃馨', 'colors': ['粉红色','白色','红色','紫色','黄色']}"

parser = PydanticOutputParser(pydantic_object=Flower)
# parser.parse(mis_formatted)

from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.fix import OutputFixingParser

new_parser = OutputFixingParser.from_llm(llm=ChatOpenAI(), parser=parser)

print('mis_formatted:', mis_formatted)
instructions = new_parser.get_format_instructions()
print('instructions:', instructions)

result = new_parser.parse(mis_formatted)
print('result:', result)
