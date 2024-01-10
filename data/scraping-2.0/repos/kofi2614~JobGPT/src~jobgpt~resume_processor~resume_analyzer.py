import json
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from jobgpt.resume_processor.resume_reader import ResumeReader
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from jobgpt.utils.llm import count_tokens, load_model
from typing import Dict
from jobgpt.utils.dataclasses import SummarySection, SkillsSection, WorkExperienceSection, EducationSection, PersonalProjectSection


system_template = """
You are an experienced career consultalt who helps clients to improve their resumes.
When you are asked to provide evaluation or suggestion, make sure your are critical and specific.
Focus on the use of professional language and the relevancy to the job description.
REMEMBER DO NOT make things up or create fake experiences. 
""".strip()

skills_teamplate = """
You are a experienced career consultalt helping clients with the skills section of their resumes given a job description that the client is applying for.
Let's think step by step and experience by experience.
First, give an CRITICAL evaluation of the skills section focusing on use of professional language and the relevancy to the job description.
Second, provide suggestions on how the client can improve the section. Mention the exact wording used and how the client can reword it. 
Try to find all the problems and give specific suggestions.
Last, give a revision of the skills section besed on your suggestions.

{section}: {section_text}
Job Description: {job_description}
""".strip()

work_experience_teamplate = """
You are a experienced career consultalt helping clients with the work experience section of their resumes given a job description that the client is applying for.
Let's think step by step and experience by experience.
First, give an CRITICAL evaluation of the work experience section focusing on use of professional language and the relevancy to the job description.
Second, provide suggestions on how the client can improve the section. Mention the exact wording used and how the client can reword it. 
Try to find all the problems and give specific suggestions.
Last, give a revision of the work experience section besed on your suggestions.

{section}: {section_text}
Job Description: {job_description}
""".strip()

education_teamplate = """
You are a experienced career consultalt helping clients with the education section of their resumes given a job description that the client is applying for.
Let's think step by step and experience by experience.
First, give an CRITICAL evaluation of the education section focusing on use of professional language and the relevancy to the job description.
Second, provide suggestions on how the client can improve the section. Mention the exact wording used and how the client can reword it. 
Try to find all the problems and give specific suggestions.
Last, give a revision of the education section besed on your suggestions.

{section}: {section_text}
Job Description: {job_description}
""".strip()

summary_teamplate = """
You are a experienced career consultalt helping clients with the summary section of their resumes given a job description that the client is applying for.
Let's think step by step and experience by experience.
First, give an CRITICAL evaluation of the summary section focusing on use of professional language and the relevancy to the job description.
Second, provide suggestions on how the client can improve the section. Mention the exact wording used and how the client can reword it. 
Try to find all the problems and give specific suggestions.
Last, give a revision of the summary section besed on your suggestions.

{section}: {section_text}
Job Description: {job_description}
""".strip()

personal_project_teamplate = """
You are a experienced career consultalt helping clients with the personal project section of their resumes given a job description that the client is applying for.
Let's think step by step and experience by experience.
First, give an CRITICAL evaluation of the personal project section focusing on use of professional language and the relevancy to the job description.
Second, provide suggestions on how the client can improve the section. Mention the exact wording used and how the client can reword it. 
Try to find all the problems and give specific suggestions.
Last, give a revision of the personal project section besed on your suggestions.

{section}: {section_text}
Job Description: {job_description}
""".strip()

section_title_map = {
    "Skills": "skills",
    "Work Experience": "work_experience",
    "Education": "education",
    "Summary": "summary",
    "Personal Projects": "personal_project"    
}
prompt_map = {
    "skill": skills_teamplate,
    "work_experience": work_experience_teamplate,
    "education": education_teamplate,
    "summary": summary_teamplate,
    "personal_project": personal_project_teamplate
}

section_model_map = {
    "skill": SkillsSection,
    "work_experience": WorkExperienceSection,
    "education": EducationSection,
    "summary": SummarySection,
    "personal_project": PersonalProjectSection    
}

class ResumeAnalyzer:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = load_model(model_name)        
        self.system_prompt = SystemMessagePromptTemplate.from_template(system_template.strip())        
    async def analyze(self, section_title: str, section_text: str, job_description: str) -> dict:                
        user_prompt = HumanMessagePromptTemplate.from_template(prompt_map[section_title])
        resume_analyzer_prompt = ChatPromptTemplate(input_variables=["section", "section_text", "job_description"], messages=[self.system_prompt, user_prompt])
        chain_analyze = LLMChain(llm=self.llm, prompt=resume_analyzer_prompt)
        analysis = await chain_analyze.arun(
            {                   
                "section": section_title, 
                "section_text": section_text, 
                "job_description": job_description
            })  
        output = {"title": section_title, "analysis": analysis}              
        return output