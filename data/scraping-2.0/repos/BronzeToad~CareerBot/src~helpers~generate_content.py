import configparser
import os
from typing import Optional

import openai
import tiktoken
from dotenv import load_dotenv

from src.helpers import toad_tools as ToadTools
from src.helpers.toad_tools import FileType
from src.helpers.create_folders import get_folder_name

# =========================================================================== #

def get_openai_api_key():
    """Retrieve the OpenAI API key from the .env file."""

    load_dotenv()
    api_key = os.getenv("openai_api_key")

    if api_key is None:
        raise Exception("OpenAI API key not found in .env file.")

    return api_key


def get_default_model():
    config = configparser.ConfigParser()
    config.read('/Users/ajp/Documents/Projects/CareerBot/main.ini')
    return config.get('OpenAI', 'Model')


def get_token_count(input_str: str, engine: Optional[str] = None):
    """Return the number of tokens in the input string."""
    _model = engine or get_default_model()
    encoding = tiktoken.encoding_for_model(_model)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def get_root_dir():
    config = configparser.ConfigParser()
    config.read('/Users/ajp/Documents/Projects/CareerBot/main.ini')
    return config.get('Directories', 'ProjectRoot')


def get_company_profile(filename: str):
    return ToadTools.get_file(
        folder=os.path.join(get_root_dir(), 'company_profiles'),
        filename=filename,
        file_type=FileType.TXT
    )

def get_job_description(filename: str):
    return ToadTools.get_file(
        folder=os.path.join(get_root_dir(), 'job_descriptions'),
        filename=filename,
        file_type=FileType.TXT
    )

def get_resume(filename: str):
    return ToadTools.get_file(
        folder=os.path.join(get_root_dir(), 'resumes'),
        filename=filename,
        file_type=FileType.TXT
    )

def generate_content_api_call(
    company: str,
    position: str,
    job_desc_filename: Optional[str] = None,
    company_filename: Optional[str] = None,
    resume_filename: Optional[str] = None,
    model: Optional[str] = None
):
    company_fname = company_filename or company.lower()
    job_desc_fname = job_desc_filename or get_folder_name(company, position)
    openai.api_key = get_openai_api_key()
    _model = model or get_default_model()
    response = openai.ChatCompletion.create(
        model=_model,
        messages=[
            {"role": "system", "content": build_system_msg(company, position)},
            {"role": "user", "content": build_resume_msg(resume_filename)},
            {"role": "user", "content": build_job_desc_msg(job_desc_fname)},
            {"role": "user", "content": build_company_profile_msg(company_fname)},
            {"role": "user", "content": "You should have everything you need "
                                        "to complete the assigned tasks."}
        ]
    )
    # print(response)
    return response.choices[0].message.content


def build_system_msg(company: str, position: str) -> str:
    system_msg = ("You are an expert career coach and writer who is helping "
                  "a client complete the following tasks:\n\n")

    system_msg += ("1. Create a compelling cover letter that explains why I am "
                   f"the best fit for the {position} position at {company}. "
                   f"Write the cover letter using the StoryBrand Framework.\n")

    system_msg += ("2. Compose a professional objective statement "
                   "demonstrating how my abilities align with the requirements "
                   f"for the {position} position at {company}.\n")

    system_msg += ("3. Provide the 10 most important skills for the "
                   f"{position} position at {company}.\n\n")

    system_msg += "Additional Instructions:\n"

    system_msg += ("- Do not use placeholders (e.g. 'XYZ Company', "
                   "'relevant skill/experience' or 'Company Name'). "
                   "Instead, use the information provided in this chat.\n")

    system_msg += ("- Strictly adhere to the information contained on the "
                   "provided resume when writing about my work history, "
                   "skills, and experience.\n")

    system_msg += ("- Utilize the information provided in the job description "
                   "and company profile where appropriate.\n\n")

    system_msg += ("I will provide the resume, the job description, and the "
                   "company profile in the messages immediately following "
                   "this message.")

    return system_msg

def build_resume_msg(resume_filename: Optional[str] = None) -> str:
    fname = resume_filename or 'resume_short'
    msg = f"---- Resume ----\n\n"
    msg += get_resume(fname)
    return msg

def build_job_desc_msg(job_desc_filename: str) -> str:
    msg = f"---- Job Description ----\n\n"
    msg += get_job_description(job_desc_filename)
    return msg

def build_company_profile_msg(company_filename: str) -> str:
    msg = f"---- Company Profile ----\n\n"
    msg += get_company_profile(company_filename)
    return msg


# =========================================================================== #

if __name__ == "__main__":

    pass
