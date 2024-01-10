from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")


if __name__ == "__main__":
    parser = PydanticOutputParser(pydantic_object=Recipe)
    format_instructions = parser.get_format_instructions()

    template = """料理のレシピを考えてください。
    
{format_instructions}

料理名：{dish}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["dish"],
        partial_variables={"format_instructions": format_instructions}
    )

    formatted_prompt = prompt.format(dish="カレー")

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    messages = [HumanMessage(content=formatted_prompt)]
    output = chat(messages)

    recipe = parser.parse(output.content)
    print(type(recipe))
    print(recipe)
