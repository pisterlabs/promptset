import re
import openai
import ast 
from pydantic import BaseModel
import streamlit as st
import json
    
def extract_keywords_from_text(text):

    # Authenticate to OpenAI
    openai.api_key = "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"
    

    # Define the prompts for GPT-3
    prompt = (f"Please extract the following categories from the job description:\n\n"
               f"- Soft Skills\n"
               f"- Technical Skills\n"
               f"- Experience\n"
               f"- Responsibilities\n"
               f"Text: {text}\n"
               'Return always a valid JSON object without any explanations and use curly brackets and double quotes. Return always only a JSON in the form of { "soft_skills": [...], "technical_skills": [...], "experience": [...], "responsibilities":[...]}\n')


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a  professional resume checker."},
                {"role": "user", "content": prompt}
            ]
        )

    res = response.choices[0].message.content
    res = json.loads(res)
    return res

def measure_overlap_dict_values(dict1, dict2):
    # Authenticate to OpenAI
    openai.api_key = "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"

    # Define the prompt for GPT-3
    prompt = (f"Return always a valid JSON object without any explanations and use curly brackets and double quotes. Using Natural Language understanding please measure the overlap percentage in 100 between the values matching keys for these two dictonaries:\n\n"
              f"Dict1: {dict1}\n"
              f"Dict2: {dict2}\n"
              'Return always only a JSON in the form of { "soft_skills": percentage, "technical_skills": percentage, "experience":percentage, "responsibilities": percentage}\n')
    
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a  professional resume checker."},
                {"role": "user", "content": prompt}
            ]
        )

    res = response.choices[0].message.content
    #st.write("response.choices[0].text.strip()", res)
    return json.loads(res)

def re_write_work_experience(work_experience, kws):
    openai.api_key = "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"
    job_title = "Data Analyst" 
    
    soft_skills = kws["soft_skills"]
    tech_skills = kws["technical_skills"]
    experience = kws["experience"]
    responsibilities = kws["responsibilities"]
                      
    # Define the prompt for GPT-3
    prompt = (f'I provide you list of my work experience: {work_experience}\n please Rephrase my each of my work experience to incorporate following keywords. Also begin with a strong action verb and use also quantitative adverb for each work expereince. For each item of list create only one rewritten text:\n\n'
              f"soft skills: {soft_skills}\n"
              f"tech skills: {tech_skills}\n"
              f"experience: {experience}\n"
              f"responsibilities: {responsibilities}\n"
              'Return always a valid JSON object without any explanations and use curly brackets and double quotes. Return always only a JSON in the form of { "re_written_work_experience": [...] }\n')
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a  professional resume checker."},
                {"role": "user", "content": prompt}
            ]
        )


    res = response.choices[0].message.content
    #st.write("response.choices[0].text.strip()", res)
    text =  json.loads(res)
    return text["re_written_work_experience"]
