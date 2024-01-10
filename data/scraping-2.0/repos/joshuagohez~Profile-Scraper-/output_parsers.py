from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Summary(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: List[str] = Field(description="Interesting facts about the person")

    def to_dict(self):
        return {
            "summary": self.summary, 
            "facts": self.facts    
        }
    
class Interest(BaseModel):
    topics_of_interest: List[str] = Field(description="Topics that may interest the person")

    def to_dict(self):
        return {
            "topics_of_interest": self.topics_of_interest
        }
    
class IceBreaker(BaseModel):
    ice_breakers: List[str] = Field(description="Create ice breakers to open a conversation with the person")

    def to_dict(self):
        return {
            "ice_breakers": self.ice_breakers
        }
    
    
summary_parser = PydanticOutputParser(pydantic_object=Summary)
interest_parser = PydanticOutputParser(pydantic_object=Interest)
icebreaker_parser = PydanticOutputParser(pydantic_object=IceBreaker)


    