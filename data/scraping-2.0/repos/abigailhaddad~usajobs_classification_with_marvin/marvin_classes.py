# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:15:02 2023

@author: abiga
"""
import os
from enum import Enum
import openai
from marvin import ai_classifier, ai_fn, settings, ai_model, settings
import logging
from typing import List
from pydantic import BaseModel

settings.llm_max_tokens=1500
llm_max_context_tokens=2500
settings.llm_temperature=0.0


#logging.basicConfig(level=logging.DEBUG) # if you want to see the JSON getting passed to OpenAI
#logging.disable(logging.CRITICAL) # if you want to turn off the logging

openai.api_key = os.environ.get("OPENAI_API_KEY")
settings.llm_model='openai/gpt-3.5-turbo'


@ai_classifier
class JobCategory(Enum):
    """Represents general areas of job responsibilities"""
    DATA_ENGINEERING_SOFTWARE = "Data Engineering and Software Development"
    DATA_SCIENCE = "Data Science"
    DATA_ANALYSIS_BI = "Data Analysis and Business Intelligence"
    RESEARCH_SCIENCE = "Research and Science"
    MANAGEMENT_LEADERSHIP = "Management and Leadership"
    
@ai_fn
def generate_job_title(duties: str) -> str:
    """Given `duties`, generates a specific job title based on the content, not based on any titles contained in the text."""


@ai_model(instructions='Extract programming languages and named software tools from the given text. Only return items directly supported in text.')
class TechDetails(BaseModel):
    text: str
    programming_languages: List[str]
    software_tools: List[str]

class JobAnalyzer:
    def __init__(self, duties: str):
        self.duties = duties
        self.category = JobCategory(self.duties).value
        self.job_title = generate_job_title(self.duties)

        # Extract tech details using the TechDetails model
        tech_details = TechDetails(self.duties)
        self.programming_languages = tech_details.programming_languages
        self.software_tools = tech_details.software_tools

    def __repr__(self):
        return (f"Duties: {self.duties}\n"
                f"Category: {self.category}\n"
                f"Job Title: {self.job_title}\n"
                f"Programming Languages: {self.programming_languages}\n"
                f"Software Tools: {self.software_tools}")


@ai_classifier
class JobTitleContrast(Enum):
    """
    Compare two job titles:
    
    If the titles are the same or one title is an obvious subset of the other, classify as "Similar".
    For example:
    - "Data Scientist" vs "Data Scientist, ZP-1560-II/III, BEA-MP-VFP" are considered similar because the core title is the same.
    - "Chief Data Officer" vs "Data Officer" are also similar.
    - "Data Analyst" vs "Business Intelligence Analyst" are similar.
    
    If the core roles or designations in the titles differ significantly, classify as "Different".
    For example:
    - "Data Analyst" vs "Data Scientist" are different as they represent distinct roles.
    """
    Similar = "Position titles are identical or extremely similar"
    Different = "Position titles are mismatched"

class TitleContraster:
    def __init__(self, job_title: str, official_title: str):
        self.job_title = job_title
        self.official_title = official_title
        
        # Create a combined string for classification
        combined_title_info = f"Generated Title: {self.job_title} | Official Title: {self.official_title}"
        
        # Classify the combined string using the JobTitleContrast classifier
        self.mismatch_level = JobTitleContrast(combined_title_info).value

    def __repr__(self):
        return f"Generated Job Title: {self.job_title}\nPosition Title: {self.official_title}\nMismatch Level: {self.mismatch_level}"



def test_functions():
    # List of duties for testing
    duties_samples_list = [
    # Mixed with software tools
    "Assist in developing data set processes. | Assist in identifying ways to improve data reliability, efficiency and quality with the use of programming language and tools. | Assist in building data visualization products. | Assist in developing mathematical and/or statistical models for evaluation, identification, collection, and/or analysis of all data used to support assigned projects or programs.",

    # Not fitting any labels
    "Construct data pipelines using complex tools and techniques to handle data at scale. | Plan and conduct relevant research and analytical studies of Army problems to support critical operational problems and/or decisions. | Develop training, marketing, and other tools as a means of increasing knowledge of data management and data analysis within the MCoE. | Develop data set processes and uses programming language and tools to identify ways to improve data reliability, efficiency and quality for missions, goals, and future planning.",

    # Explicit software mention
    "Duties are listed at the full performance level. | Performs ad-hoc data mining and exploratory statistics tasks on very large datasets. | Utilizes general purpose programming libraries to complete tasks related to data science or modeling. | Develop reports that translate technical analyses to concise and understandable information. | Develop proposals for the initiation of additional data studies or analyses. | Utilize computer programs to create statistical or mathematical models."]

    # Test with the provided JobAnalyzer class
    for duties in duties_samples_list:
        analysis = JobAnalyzer(duties)
        print(analysis)
        print("-" * 80)


import os
import openai
from marvin import ai_fn, settings, settings
from typing import List, Dict
from pydantic import BaseModel

settings.llm_max_tokens=1500
llm_max_context_tokens=2500
settings.llm_temperature=0.0


#logging.basicConfig(level=logging.DEBUG) # if you want to see the JSON getting passed to OpenAI
#logging.disable(logging.CRITICAL) # if you want to turn off the logging

openai.api_key = os.environ.get("OPENAI_API_KEY")
settings.llm_model='openai/gpt-3.5-turbo'

@ai_fn
def fetch_related_tasks(base_prompt: str) -> Dict[str, int]:
    """Given `task`, generate a dictionary of ten tasks that are related to that task. The keys are the tasks, and the values are numbers from 1-10, indicating the degree of relatedness to the prompt."""


class BaseTask:
    def __init__(self, task: str):
        self.task = task
        self.related_tasks = fetch_related_tasks(self.task)

    def __repr__(self):
        return f"Base Prompt: {self.task}\nRelated Tasks: {self.related_tasks}"


# Using the BasePrompt class
base_prompt_obj = BaseTask("Data analysis")
print(base_prompt_obj)