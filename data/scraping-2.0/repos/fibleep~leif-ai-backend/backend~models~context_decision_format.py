from langchain_core.pydantic_v1 import BaseModel
from pydantic import Field


class ContextDecisionFormat(BaseModel):
    is_about_location: bool = Field(..., description="Is the user talking about a "
                                                     "location?")

    class Config:
        arbitrary_types_allowed = True
