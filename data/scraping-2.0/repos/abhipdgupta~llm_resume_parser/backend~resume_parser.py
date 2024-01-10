from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
import json
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from propmts import PROMPT_4
from dotenv import load_dotenv

load_dotenv()
from typing import List, Optional,Union,Any
from pydantic import BaseModel, Field


class ContactInformation(BaseModel):
    Name: Optional[str] = None
    Email: Optional[str] = None
    Contact: Optional[str] = None
    Links: Optional[List[str]] = None


class WorkExperience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None


class Education(BaseModel):
    course: Optional[str] = None
    branch: Optional[str] = None
    institute: Optional[str] = None


class Projects(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None
    
class OutputFormat(BaseModel):
    Contact_Information: Optional[Any] = None
    About_Me: Optional[Any] = None
    Work_Experience: Optional[Any] = None
    Education: Optional[Any] = None
    Skills: Optional[Any] = None
    Certificates: Optional[Any] = None
    Projects: Optional[Any] = None
    Achievements: Optional[Any] = None
    Interests:Optional[Any]=None
    Volunteer: Optional[Any] = None

# class OutputFormat(BaseModel):
#     Contact_Information: Optional[ContactInformation] = None
#     About_Me: Optional[str] = None
#     Work_Experience: Optional[Union[None, List[WorkExperience]]] = None
#     Education: Optional[Union[None, List[Education]]] = None
#     Skills: Optional[List[str]] = None
#     Certificates: Optional[List[str]] = None
#     Projects: Optional[Union[None, List[Projects]]] = None
#     Achievements: Optional[List[str]] = None
#     Volunteer: Optional[List[str]] = None


def check_job_compatibility(parsed_resume: str):
    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")

    template = """check that the give resume {parsed_resume}
    
    is suitable for full stack developer role:
    
    """

    prompt_template_name = PromptTemplate(
        input_variables=["parsed_resume"],
        template=template,
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

    response = name_chain(inputs={"parsed_resume": parsed_resume})
    print(response["text"])


def ExtractInformationFromResume(resume: str) -> OutputFormat:
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-1106")
    template = PROMPT_4
    parser = PydanticOutputParser(pydantic_object=OutputFormat)
    prompt_template_name = PromptTemplate(
        input_variables=["resume"],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

    response = name_chain(inputs={"resume": resume})
    result=parser.parse(response["text"]).json()
    return result


def PdfToText(doc_path: str) -> str:
    loader = PyPDFLoader(doc_path)
    pages = loader.load_and_split()

    return pages[0].page_content


if __name__ == "__main__":
    extracted_text = PdfToText("Abhishek Resume.pdf")
    info = ExtractInformationFromResume(extracted_text)
    # with open("./parsed_resume/Abhishek Resume.json", "r") as f:
    #     print(json.load(f))
