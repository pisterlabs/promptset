from typing import List, Optional
from openai_function_call import OpenAISchema

class Address(OpenAISchema):
    street: str
    city: str
    state: str
    zip: str

class PersonalInformation(OpenAISchema):
    name: str
    email: str
    phone: str
    address: Address

class Education(OpenAISchema):
    degree: str
    institution: str
    year: int

class WorkExperience(OpenAISchema):
    position: str
    company: str
    startDate: str
    endDate: Optional[str]
    responsibilities: List[str]

class Language(OpenAISchema):
    language: str
    proficiency: str

class Project(OpenAISchema):
    title: str
    description: str
    technologies: List[str]

class CandidateRecord(OpenAISchema):
    personalInformation: PersonalInformation
    education: List[Education]
    workExperience: List[WorkExperience]
    skills: List[str]
    languages: List[Language]
    projects: List[Project]