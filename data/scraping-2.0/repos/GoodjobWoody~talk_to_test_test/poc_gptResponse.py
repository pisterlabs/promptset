import json
import logging
import os
import openai
from dotenv import load_dotenv
import PyPDF2

os.environ['REQUESTS_CA_BUNDLE'] = './open_ai_cert.cer'
load_dotenv()  # take environment variables from .env.

logger = logging.getLogger("My Logger")
logger.setLevel(logging.DEBUG)


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Check if the PDF is encrypted
        if pdf_reader.is_encrypted:
            pdf_reader.decrypt('')

        # Extract text from each page
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            # page = pdf_reader.getPage(page_num)
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


# read resume file
resume_path = "./Resume_Loren_Wang_CA.pdf"

# resume_content = extract_text_from_pdf(resume_path)
with open('resume_sample.json', 'r') as file:
    resume_content = json.load(file)

# Now, 'data' contains the content of the JSON file as a Python object
# read job description


jd_path = "./job_description_3.txt"
with open(jd_path, 'r') as file:
    jd_content = file.read()
    print(jd_content)

# Remove line spaces
modified_content = jd_content.replace('\n', ' ')

# Write the modified content back to the file
with open(jd_path, 'w') as file:
    file.write(modified_content)

# Write the modified content back to the file
with open('keywords.json', 'r') as file:
    keywords = file.read()

# call api

# input_message = {
#   'resume': resume_content,
#   'job_description': jd_content
# }

input_message = {
    "highlights": resume_content,
    "keywords": keywords}

# input_message = jd_content

print("***********************************")
openai.api_key = os.getenv("OPENAI_API_KEY")

promptJavaCoding = "You are a very knowledgeable and senior Python developer. First, provide complete code solution with minimum time complexisty and using Python built-in methods as much as possible; Second, give a short explaination of the algorithm; Thirdlly, give the time complexity, like O(1), O(n), etc "
promptJavaQAquestion = "You are a very knowledgeable and senior Java developer. You shall answer the questions with concise exmplaination "
# prompt_convert_resume_to_json = "Please convert the input resume into json format"
prompt_resume_tailoring = " Plesae tailor the input resume to match required keywords as many as possible. Please return the revised resume in a json format. "
prompt_convert_resume_to_json = "please convert this resume to joson format"
prompt_revise_highlights_of_resume = "plesae tailor the resume highlights according to requirements of job description. No more than six bulletpoints. No more than 20 words each bulletpoint. List out matching keywords in job description "
prompt_revise_experience_of_resume = "plesae tailor the all of job responsibilities in resume, according to requirements of job description. And return in json format"
prompt_extract_keywords_from_jd = "extract and list all keywords in the job description. Categorize the keywords, as hardskills, soft skills, certificates, education,etc. Return in json format "
# logger.error(openai.Model.list())

print("***********************************")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_resume_tailoring},
        {"role": "user", "content": str(input_message)}
    ]
)

logger.error(completion.choices[0].message.content)
