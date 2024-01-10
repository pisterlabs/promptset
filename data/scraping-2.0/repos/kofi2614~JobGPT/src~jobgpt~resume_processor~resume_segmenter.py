from jobgpt.utils.dataclasses import SegmentedResume
import os
import json
from jobgpt.utils.llm import count_tokens, load_model
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

parser = PydanticOutputParser(pydantic_object=SegmentedResume)
system_template = """
You are an experienced career consultalt who helps clients to improve their resumes.
You should be familiar with the general structure of a resume and the use of professional language.
When you are asked to provide evaluation or suggestion, make sure your are critical and specific.
Focus on the use of professional language and the relevancy to the job description.
REMEMBER DO NOT make things up or create fake experiences. 
""".strip()

user_teamplate = """
You are an experienced career consultalt helping clients with their resumes.
First, let's understand the client's background by reading the resume. 
Your job is to read the resume and segment the resume into different sections.
A general resume should at least have work experience and education sections. 
It may also have additional sections such as personal projects, summary and skills.
Segment the given resume into the sections mentioned above.
If you think that one section is missing, just DO NOT include the segment key in the output
{json_format}

resume: {resume_text}
""".strip()

user_teamplate = """
You are an experienced career consultalt helping clients with their resumes.
First, let's understand the client's background by reading the resume. 
Your job is to read the resume and segment the resume into different sections.
The sections of resume should fall into one of the categories:
[work experience, education, personal projects, summary and skills.]
Segment the given resume into the sections mentioned above.
Give your output in JSON format where the keys are the section names and the values are the content of the sections.
The section content should be pure text. 
Only use the content from the resume. DO NOT add any additional information.
Format the section content text by adding line breaks so it can be printed nicely
{json_format}

resume: {resume_text}
""".strip()
class ResumeSegmenter:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = load_model(model_name=model_name)
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        user_prompt = HumanMessagePromptTemplate.from_template(user_teamplate)
        resume_segmenter_prompt = ChatPromptTemplate(input_variables=["json_format", "resume_text"], messages=[system_prompt, user_prompt])
        self.chain_segmenter= LLMChain(llm=self.llm, prompt=resume_segmenter_prompt)
    def segment(self, resume_text: str):        
        segmented_resume = self.chain_segmenter.run({
                "resume_text": resume_text, 
                "json_format": parser.get_format_instructions()
            })        
        segmented_resume = json.loads(segmented_resume)
        try: 
            SegmentedResume(**segmented_resume)
        except Exception as e:
            print(e)
            print(segmented_resume)            
        return segmented_resume
        