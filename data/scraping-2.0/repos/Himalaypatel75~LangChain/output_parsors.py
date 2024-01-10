from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class PersonIntel(BaseModel):
    summary: str = Field(description="Summery Of the person")
    facts: List[str] = Field(description="Facts about the person")
    topic_of_interest: List[str] = Field(
        description="Topic that may interest the person"
    )
    ice_breakers: List[str] = Field(
        description="Create ice breakers to open a conversation with the person"
    )

    def to_dict(self):
        return {
            "summery": self.summary,
            "facts": self.facts,
            "topic_of_interest": self.topic_of_interest,
            "ice_breakers": self.ice_breakers,
        }


person_intel_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=PersonIntel
)
