import os
import re
import json
import openai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import PyPDF2
from pdfminer.high_level import extract_text

# env 
load_dotenv()
skills_db = pd.read_csv("skills.csv")
openai.api_key = os.getenv("OPENAI_API_KEY")
gpt_model = os.getenv("GPT_MODEL")

# gpt_part
system_prompt = "You are a resume parser. Your job will be to extract data from the resume into JSON format as per the syntax and instructions below.\n[output only JSON]\n[Formatting Instructions]\n====================\n{\"name\",\n\"phone_number\",\n\"email\",\n\"github_profile_link\",\n\"linkedin_profile_link\",\n\"field_of_study\",\n\"education\": [{\"degree\", \"institute\", \"year\", \"gpa\"}],\n\"achievements\",\n\"skills\": [] (You have to go through whole of resume and extract all possible and relevant skills from it),\n\"relevant_courses\": [],\n\"projects\": [{\"name\", \"organisation\", \"timeline\", \"brief_description\", \"project_link\"}],\n\"position_of_responsibilities\": [{\"position\", \"organisation\", \"tenure\", \"brief_description\"}],\n\"summary\": (You have to summarize the resume summing up all the relevant skills and qualities of the individual within 50 words)}\n===================="
def get_resume(user_prompt, temp, output_limit):
    response = openai.ChatCompletion.create(
        model = gpt_model,
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature = temp,
        max_tokens = output_limit
    )

    print(response)
    return response.choices[0].message["content"]

# generating usr_input from resume
def generate_prompt(FILEPATH):
    txt = extract_text(FILEPATH)
    PDFFile = open(FILEPATH,'rb')
    PDF = PyPDF2.PdfReader(PDFFile)
    pages = len(PDF.pages)
    key = '/Annots'
    uri = '/URI'
    ank = '/A'

    links = []
    for page in range(pages):
        print("Current Page: {}".format(page))
        pageSliced = PDF.pages[page]
        pageObject = pageSliced.get_object()
        if key in pageObject.keys():
            ann = pageObject[key]
            for a in ann:
                u = a.get_object()
                if uri in u[ank].keys():
                    links.append(u[ank][uri])
    link = "\n".join(links)

    prompt = txt + "\n" + link
    return prompt

# resume analysis 
def get_skill_score(resume_data):
    if (type(resume_data) == str):
        resume_data = json.loads(resume_data)
    user_skills = resume_data["skills"]

    score = 0
    for s in user_skills:
        fnd = skills_db["skill_name"][skills_db["skill_name"] == s.lower()].index
        if (len(fnd) > 0):
            score += skills_db["skill_rank"][fnd[0]]
        else:
            score += 5
    score /= max(len(user_skills), 1)
    score *= (np.log(min(max(3, len(user_skills)), 20)) / np.log(20))

    # Out of 100
    return score * 10

def score_resume(resume_data):
    # 40% weightage to skills, 40% weightage to acads, 20% weightage to completeness of resume
    f_score = 0
    if (type(resume_data) == str):
        resume_data = json.loads(resume_data)
    
    # a. Skills shit
    s_score = get_skill_score(resume_data)
    f_score += 0.4 * s_score

    # b. Completeness
    c_score = 0
    for x in resume_data.keys():
        d = resume_data[x]
        if (type(d) == str or type(d) == list or type(d) == dict):
            c_score += (len(d) != 0)
        else:
            c_score += 1
    c_score /= max(len(resume_data.keys()), 1)
    c_score *= 100
    f_score += 0.2 * c_score

    # c. Acads
    a_score = 0
    p_ptrn = r"((?:\d{1,2}(?:\.\d{1,2})?|100)(?:\s*%))"
    g_ptrn = r"((?:\d{1,2}(?:\.\d{1,2})?|100)(?:\s*\/\s*\d+))"
    for edu in resume_data["education"]:
        g = re.findall(g_ptrn, edu["gpa"])
        p = re.findall(p_ptrn, edu["gpa"])
        if (len(g) != 0):
            g = g[0].split("/")
            a_score = (float(g[0]) / float(g[1])) * 100
            if a_score != 0:
                break
        if (len(p) != 0):
            p = p[0].split("%")
            a_score = float(p[0])
            if a_score != 0:
                break
    if a_score == 0:
        a_score = 50
    f_score += 0.4 * a_score

    return s_score, c_score, a_score, f_score