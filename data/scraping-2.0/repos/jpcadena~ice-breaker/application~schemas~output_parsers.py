"""
A module for output parsers in the application-schemas package.
"""
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str = Field(title="Summary", description="summary")
    facts: list[str] = Field(description="interesting facts about them")


class IceBreaker(BaseModel):
    ice_breakers: list[str] = Field(
        title="IceBreaker", description="ice breaker list"
    )


class TopicOfInterest(BaseModel):
    topics_of_interest: list[str] = Field(
        title="Topic Of Interest",
        description="topic that might interest the person"
    )


summary_parser: PydanticOutputParser = PydanticOutputParser(  # type: ignore
    pydantic_object=Summary)
ice_breaker_parser: PydanticOutputParser = PydanticOutputParser(  # type: ignore
    pydantic_object=IceBreaker)
toi_parser: PydanticOutputParser = PydanticOutputParser(  # type: ignore
    pydantic_object=IceBreaker)
