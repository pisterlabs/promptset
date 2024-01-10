from jobtractor.models import JobData
from langchain.chat_models import ChatLiteLLM
from dagster import op
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import re

load_dotenv()

llm = ChatLiteLLM(model="perplexity/mistral-7b-instruct", temperature=0.0)

@op()
def text_extract_single(content: str):

    parser = PydanticOutputParser(pydantic_object=JobData)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "Please answer the following query as specified and do not add additional information. \n\n{format_instructions}\n\n{query}\n"
            )
        ],
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    _input = prompt.format_prompt(query=content)
    output = llm(_input.to_messages())

    output_content = output.content.replace("\\_", "_").replace("}}","}")

    output_content = output_content.replace("null", "None")

    if output_content[-1].isdigit():
        output_content += "\n}"

    if output_content[-1].isalpha():
        output_content += '"\n}'

    parsed = eval(output_content)

    data = JobData(
        job_name=parsed["job_name"],
        job_location=parsed["job_location"],
        job_required_years_work_experience=parsed["job_required_years_work_experience"],
    )
   
    return data

