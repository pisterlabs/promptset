from langchain_core.pydantic_v1 import BaseModel, Field


class CodeHelper(BaseModel):
    code: str = Field(description="shell script to run")
    description: str = Field(description="What does the code do")