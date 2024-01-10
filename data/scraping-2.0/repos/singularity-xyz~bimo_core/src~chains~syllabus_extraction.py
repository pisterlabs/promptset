from .extraction import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class Assignment(BaseModel):
    id: Optional[str]
    name: Optional[str]
    description: Optional[str]
    due_date: Optional[str]
    weightage: Optional[float]

class Exam(BaseModel):
    id: Optional[str]
    name: Optional[str]
    description: Optional[str]
    date: Optional[str]
    weightage: Optional[float]

class Syllabus(BaseModel):
    id: Optional[str]
    course_code: Optional[str]
    course_name: Optional[str]
    assignments: Optional[List[Assignment]] = []
    exams: Optional[List[Exam]] = []


class SEChain:
    def __init__(self, schema: BaseModel = Syllabus, llm: ChatOpenAI = ChatOpenAI(model="gpt-3.5-turbo-16k", verbose=True)):
        self.chain = create_extraction_chain_pydantic(llm=llm, pydantic_schema=schema)

    def __getattr__(self, name):
        # If the attribute isn't found in SyllabusExtractionChain, look for it in self.chain
        return getattr(self.chain, name)
