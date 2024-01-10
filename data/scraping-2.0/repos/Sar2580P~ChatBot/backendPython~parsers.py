from langchain.output_parsers import *
from pydantic import validator
from pydantic import BaseModel, Field
from typing import List


# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least 3 bullet points
    # allow_reuse : whether to track and raise an error if another validator refers to decorated function
    @validator('summary', allow_reuse=True)   
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines

# set up output parser
summary_parser = PydanticOutputParser(pydantic_object=ArticleSummary)