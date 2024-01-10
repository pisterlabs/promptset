import os
import json
import openai
import numpy as np
import re
from dotenv import load_dotenv
import PyPDF2
from pdfminer.high_level import extract_text
from .models import ResumeData
import pandas as pd
# import time
# env 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
skills_db = pd.read_csv("/home/aleatoryfreak/resumate/resumate/files/codes/skills.csv")
system_function = os.getenv("SYSTEM_FUNCTION")

def get_resume(user_prompt, temp, output_limit):
    response = openai.ChatCompletion.create(
        model = gpt_model,
        messages = [
            {
                "role": "system",
                "content": system_function
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
def addresumedatatodb(filename, FILEPATH):
    txt = extract_text(FILEPATH)
    PDFFile = open(FILEPATH,'rb')
    PDF = PyPDF2.PdfReader(PDFFile)
    pages = len(PDF.pages)
    key = '/Annots'
    uri = '/URI'
    ank = '/A'

    links = []
    for page in range(pages):
        # print("Current Page: {}".format(page))
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
    response = get_resume(prompt, 0.5, 1800)
    # response = open("/home/aleatoryfreak/resumate/resumate/files/codes/vll.txt", "r").read()
    try:
        s1, s2, s3, sf = score_resume(response)  
    except Exception as e:
        s1, s2, s3, sf = 0, 0, 0, 0
    ress = '[' + response + ']'
    userdata = ResumeData(uuid=filename, data=ress, skill_score=s1, completeness_score=s2, academic_score=s3, overall_score=sf)
    userdata.save()

    return response

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
