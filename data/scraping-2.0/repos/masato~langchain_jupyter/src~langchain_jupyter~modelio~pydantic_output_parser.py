from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field, validator

chat = ChatOpenAI()


class Smartphone(BaseModel):
    release_date: str = Field(..., description="スマートフォンの発売日")
    screen_inches: float = Field(
        ...,
        description="スマートフォンの画面サイズ(インチ)",
    )
    os_installed: str = Field(
        ...,
        description="スマートフォンにインストールされているOS",
    )
    model_name: str = Field(..., description="スマートフォンのモデル名")

    @validator("screen_inches")
    def validate_screen_inches(cls, field: float) -> float:  # noqa: ANN101
        error_message = "画面サイズは0より大きい必要があります"
        if field <= 0:
            raise ValueError(error_message)
        return field


parser = OutputFixingParser.from_llm(
    llm=chat,
    parser=PydanticOutputParser(pydantic_object=Smartphone),
)


result = chat.invoke(
    [
        HumanMessage(content="Android でリリースされたスマートフォンを 1 個挙げて"),
        HumanMessage(content=parser.get_format_instructions()),
    ],
)

parsed_result = parser.invoke(result)
print(f"スマートフォンのモデル名: {parsed_result.model_name}")
print(f"スマートフォンの発売日: {parsed_result.release_date}")
print(f"スマートフォンの画面サイズ: {parsed_result.screen_inches}")
print(f"スマートフォンのOS: {parsed_result.os_installed}")
