"""Collection of LLM chains."""
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


class CVSerializationChain(LLMChain):
    def __init__(self):
        prompt = PromptTemplate.from_template("""
            Extract information from the CV provided. Return the information grouped under the following: Contact, 
            Academic Background, Professional Experience, Skills, Other. Start each group following this pattern: 
            [<GROUP NAME>], e.g. for 'Skills' do '[SKILLS]'. Return the information as detailed as possible. To 
            separate topics, use bullet points.
            CV: {cv}
        """)
        super().__init__(
            llm=OpenAI(max_tokens=2000),
            prompt=prompt,
        )


class JobPostSerializationChain(LLMChain):
    def __init__(self):
        prompt = PromptTemplate.from_template("""
            For the following HTML response, extract the following: company name, job description, company description, 
            job requirements. Start each group following this pattern: [<GROUP NAME>], e.g. for 'Skills' do '[SKILLS]'. 
            Return the information as detailed as possible. HTML response: '{content}'")
        """)
        super().__init__(
            llm=OpenAI(max_tokens=1000),
            prompt=prompt,
        )


class CoverLetterCreationChain(LLMChain):
    def __init__(self):
        prompt = PromptTemplate.from_template("""
            Based on given CV information and a job posting, write a cover letter. Write 230 to 330 words. Start with 
            'Dear Sir or Madam'. Do not use any placeholder values. Only make claims that are backed by data on the CV. 
            Focus on mentioning how the requirements are fulfilled instead of listing everything from the CV. 
            Additionally, follow these guidelines:
            [[INTRODUCTION (1st paragraph)]]
            - State clearly in your opening sentence the purpose for your letter and a brief professional introduction.
            - Specify why you are interested in that specific position and organization.
            - Provide an overview of the main strengths and skills you will bring to the role.
            
            [[BODY (2-3 paragraphs)]]
            - Cite a couple of examples from your experience that support your ability to be successful in the position or organization.
            - Try not to simply repeat your resume in paragraph form, complement your resume by offering a little more detail about key experiences.
            - Discuss what skills you have developed and connect these back to the target role.
            
            [[CLOSING (last paragraph)]]
            - Restate succinctly your interest in the role and why you are a good candidate.
            - Thank the reader for their time and consideration.
            
            In the following, the required information about the applicant and the job:
            [[CV]]
            {cv}
            
            [[JOB POSTING]]
            {job_posting}
        """)
        super().__init__(
            llm=OpenAI(max_tokens=1000),
            prompt=prompt,
        )
