import string
import random
import re
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from db import resume_details_collection
import os
OPENAIKEY=os.environ['OPENAIKEY']
llm = OpenAI(openai_api_key=OPENAIKEY,  max_tokens=-1)

template = """You are a chatbot who helps people to build their resume/portfolio. This is the HTML of the portfolio {html}. Analyze the HTML properly.The following statement "{statement}" would be an instruction or information related to skills, achievements, education, projects or any other section in the resume. Analyze the statement and update the HTML code according to statement. You are free to add or remove a section as per the scenario. Make the portfolio attractive in styling. Return me only the HTML Code.
"""
prompt = PromptTemplate(template=template, input_variables=["html", "statement"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def query_update_billbot(user_id, statement):
    resume_html = get_resume_html_db(user_id)
    html_code = llm_chain.run({"html": str(resume_html), "statement": statement}) 
    return html_code

def get_resume_html_db(user_id):
    if resume_data := resume_details_collection.find_one({"user_id": user_id}):
        resume_html = resume_data.get("resume_html")
        return str(resume_html)
    else:
        return ""

def add_html_to_db(user_id, html_code):
    resume_details_collection.update_one({"user_id": user_id},{"$set": {"resume_html": html_code}})