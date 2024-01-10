from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field


class TextParsing(BaseModel):
    summary: str = Field(description='大V个人简介')
    facts: list[str] = Field(description='大V的特点')
    interest: list[str] = Field(description='这个大V可能感兴趣的事情')
    letter: list[str] = Field(description='一篇联络这个大V的邮件')

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "facts": self.facts,
            "interest": self.interest,
            "letter": self.letter,
        }


letter_parser: PydanticOutputParser[TextParsing] = PydanticOutputParser(pydantic_object=TextParsing)
