from dotenv import load_dotenv, find_dotenv
import openai
import os
import re
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
# from resume_parser import resumeparse
# os.environ["OPENAI_API_KEY"] = ""

# with open("/home/samarthrawat1/Downloads/720CH1018_SamarthRawat_Dual_Ch_Intern_1.pdf", 'rb') as pdf:
#     pdf_reader = PdfReader(pdf)
#     text = ((pdf_reader.pages[0]).extract_text())
# print(text)
# data = text


chat = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-16k")
jd="SEO manager. High social marketing skills required"
system_prompt = f"""
You are an AI model that scores CVs.
You will be provided the data of a CV. 
The most important scoring criteria is relevance to the job description.
The job description will be provided to you.
Maximum score of the CV can be 150
IMPORTANT do not reply with "As an AI model..." under any circumstances 
IMPORTANT DO NOT REMOVE OR CHANGE ANY OF THE ORIGINAL TEXT PROVIDED TO YOU THAT ARE NOT GETTING IMPROVED
IMPORTANT Always mention a CV score and details of the candidate like name, phone, email
"""

human_message_example = f"""

Job desription ={jd}
Samarth Rawat
B.Tech.+ M.Tech.|NIT Rourkela
Pre‐Final Year, Chemical Engg.
DOB: 01October 2002
Contact: +919772359443
Email.: samarthrawat1@gmail.com
Education
2020‐PRESENT
B.TECH. + M.TECH(DUAL DEGREE), CH
NITRourkela
CGPA :7.37/10
MAY 2018
INTERMEDIATE
Emmanuel Mission School, Kota
Percentage: 92%
MAY 2016
MATRICULATION
Macro Vision Academy, Burhanpur
CGPA: 9.0/10.0%
Links
HackerRank:// samarthrawat1
Github:// samarthrawat1
LinkedIn:// samarthrawat1
Skills
GENERAL PROGRAMMING
C,C++, Python, Linux, Solidity, JavaScript,
Regex
PYTHON LIBRARIES
Django, FASTApi, Numpy, Pandas, seaborn,
plotly, json, collections, beaultifulsoup4,
selenium
OPERATING SYSTEMS
Windows, Linux
SOFTWARES
Google Cloud Console, thonny
LANGUAGES
English, Hindi
Relevant Courses
Data Structures
Algorithm
Numerical MethodsWork Experience/Projects
2023‐NOW Cloud Associate at Solving for India Hackathon
Topic: Hosting websites on GCP
Used Cloud VMs tohost afully‐fledged website
round theclock using nginx server andgunicorn for
load handling and assigning proper workers. Also
learned about different solutions.
2022‐2023 Web Scraping specialist Hackathon Project
Topic: Web Scraping
Used Selenium toobtain medical information from a
government website. Learned about using headers
andhow web drivers work.
2023‐NOW Appwrite Specialist Hackathon Project
Topic: BaaS connection
Used Appwrite’s Backend‐as‐a‐Service application
tocreate authentication, storage, database manage‐
ment, andcloud functions toconnect react front‐end
toafull‐fledged backend having python functions.
Achievements/Certifications
2021‐NOW 11 Google Cloud Certificates Google
Completed 30days ofgoogle cloud, both tracks.
2021‐NOW Automate the boring stuff Udemy
Learned how tousepyxl2, selenium, andother python mod‐
ulesforautomation.
2023‐NOW Solving for India Hackathon Regionals
Hosted awebsite onAMD instance ongoogle cloud VM.
2021‐NOW Elements of AI Certificate
certificate issued byuniversity ofHelsinki.
Extra Curricular Activities
2015‐2017 Youth hostel Association of India Member
Ihave been part oftheyouth hostel andwent onmultiple
trekks thattaught mediscipline, endurance, andteamwork.
2022‐2023 Core Team, OpenCode College Club
Iwasapart ofthecore team oftheOpen Source club of
College andhelped with theorientation offreshers inmy
2ndyear.
2020‐2023 Member, Microsoft Campus Club College Club
Ihave been apart oftheleading coding club ofthecollege
andhelped inorganizing most oftheactivities held bythe
club.
"""

AI_message_example = """
Name: Samarth Rawat
Email: samarthrawat1@gmail.com
Phone:+9197723459443

CV score - 75/150

HIGHLIGHTS: 
-> Has adequent skill based on the job description 
-> Has a good educational background
-> Has a good work experience

DEMERITS:
-> Has not worked with SQL 
-> No open source contributions
"""

def func_(CV):
    with open(CV, 'rb') as pdf:
        pdf_reader = PdfReader(pdf)
        data = ((pdf_reader.pages[0]).extract_text())
    store = chat(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message_example),
            AIMessage(content=AI_message_example),
            HumanMessage(content=jd+'\n'+data)
        ]
    )
    return store

# store = func_(data)
# print((store.content))
def separator(store):
    contents=store.content
    name_pattern = r"Name:\s*(.*)"
    email_pattern = r"Email:\s*(.*)"
    phone_pattern = r"Phone:\s*(.*)"
    cv_score_pattern = r"CV score\s*.\s*(\d+)/"
    highlights_pattern = r"HIGHLIGHTS:(.*?)DEMERITS"
    demerits_pattern = r"DEMERITS:(.*)"

    # Extracting the information using regular expressions
    name = re.search(name_pattern, contents).group(1)
    email = re.search(email_pattern, contents).group(1)
    phone = re.search(phone_pattern, contents).group(1)
    cv_score = int(re.search(cv_score_pattern, contents).group(1).strip())

    # highlights_text = re.search(highlights_pattern, contents, re.DOTALL).group(1).strip()
    # highlights = [highlight.strip() for highlight in highlights_text.split("\n->")]

    # demerits_text = re.search(demerits_pattern, contents, re.DOTALL).group(1).strip()
    # demerits = [demerit.strip() for demerit in demerits_text.split("\n->")]
    info_dict = {
        "name": name,
        "email": email,
        "phone": phone,
        "score": cv_score,
        # "highlights": highlights,
        # "demerits": demerits
    }
    return (info_dict)
# print(separator(store))