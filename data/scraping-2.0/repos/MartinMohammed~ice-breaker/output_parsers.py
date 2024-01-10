from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Represents information gathered about a person from an AI-based application.
class PersonIntel(BaseModel):
    """
    PersonIntel represents the information obtained from the AI-based application.
    It includes fields derived from the PromptTemplate defined in the ice_breaker file and
    the expected data from the language model (LLM) when receiving information from the Twitter
    and LinkedIn agents.
    """

    summary: str = Field(description="Summary of the person")
    facts: List[str] = Field(description="Interesting facts about the person")
    topics_of_interest: List[str] = Field(
        description="Topics that may interest the person"
    )
    ice_breakers: List[str] = Field(
        description="Ice breakers to initiate a conversation with the person"
    )

    def to_dict(self):
        """
        Converts the PersonIntel object to a dictionary.

        Returns:
        - Dictionary with keys as fields in PersonIntel and corresponding values.
        """
        return {
            "summary": self.summary,
            "facts": self.facts,
            "topics_of_interest": self.topics_of_interest,
            "ice_breakers": self.ice_breakers,
        }


# Definition of output parser with PydanticOutputParser for PersonIntel objects.
# It allows specifying the expected format of the results obtained from the AI application.
person_intel_parser = PydanticOutputParser(pydantic_object=PersonIntel)
