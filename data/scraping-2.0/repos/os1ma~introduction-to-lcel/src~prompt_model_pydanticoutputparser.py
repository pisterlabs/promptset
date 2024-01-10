from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")


output_parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = PromptTemplate.from_template(
    """料理のレシピを考えてください。

{format_instructions}

料理名: {dish}""",
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

model = ChatOpenAI(model="gpt-3.5-turbo-1106").bind(
    response_format={"type": "json_object"}
)

chain = prompt | model | output_parser

result = chain.invoke({"dish": "カレー"})
print(type(result))
print(result)
