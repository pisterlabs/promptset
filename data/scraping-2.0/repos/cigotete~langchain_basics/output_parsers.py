from typing import List

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class BookIntel(BaseModel):
    title: str = Field(description="Title of the book")
    suitable_topics: bool = Field(description="Boolean Suitability of the book")
    suitable_topics_details: str = Field(description="Why is suitability? please explain in a detailed manner why the book is suitable or not suitable for someone")
    
class CustomPydanticOutputParser(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        return """
        {
            "title": "string",
            "suitable_topics": "boolean",
            "suitable_topics_details": "string"
        }
        """

suitability_parser = CustomPydanticOutputParser(
    pydantic_object=BookIntel
    )