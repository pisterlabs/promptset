from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class LinkedinProfile(BaseModel):
    summary: str = Field(..., description="The summary of the person")  # ... 表示必须要有这个参数
    pic: str = Field(
        ...,
        description="The profile picture url of the person, often in profile_pic_url",
    )
    facts: List[str] = Field(description="The facts of the person")
    topics_of_interest: List[str] = Field(
        description="The topics of interest of the person"
    )
    ice_breakers: List[str] = Field(
        description="Create ice breakers to open a conversation with"
    )

    def to_dict(self):
        return {
            "summary": self.summary,
            "facts": self.facts,
            "topics_of_interest": self.topics_of_interest,
            "ice_breakers": self.ice_breakers,
            "pic": self.pic,
        }


person_intel_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=LinkedinProfile
)
