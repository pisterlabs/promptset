import fitz
import json
from openai import OpenAI
from PyPDF2 import PdfReader 
import textract
import re

def hightlight_content(texts, source_file, target_file):
    doc = fitz.open(source_file)
    
    for text in texts:
        for page in doc:
            text = text
            text_instances = page.search_for(text)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()

    doc.save(target_file, garbage=4, deflate=True, clean=True)


def get_resume_hightlight(source_file, openai_api_key):
    try:
        reader = PdfReader(source_file)
    except FileNotFoundError as e:
        print(f"The resume is not found. Please enter the correct file path.\n"
          f"{e}")
    page = reader.pages[0] 
    resume_text = page.extract_text() 
    print(resume_text)
    
    client = OpenAI(
        api_key = openai_api_key,
    ) 
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Extract highlight experiences for the resume and return original setences in a array format: " + resume_text,
            },
        ],
    )
    print(completion.choices[0].message.content)
    extract_hightlight_sentence(completion.choices[0].message.content)
    return completion.choices[0].message.content

def get_sop_hightlight(source_file, openai_api_key):
    try:
       sop_text = textract.process(source_file, method='pdfminer')
       print(sop_text)
    except FileNotFoundError as e:
        print(f"The sop is not found. Please enter the correct file path.\n"
          f"{e}")
    
    client = OpenAI(
        api_key = openai_api_key,
    ) 
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Extract highlight experiences for the sop and return original setences in a array format: " + str(sop_text),
            },
        ],
    )
    print(completion.choices[0].message.content)
    extract_hightlight_sentence(completion.choices[0].message.content)
    return completion.choices[0].message.content


def extract_hightlight_sentence(openai_output_string):
    extracted_strings = re.findall(r'"(.*?)"', openai_output_string)
    print(extracted_strings)
    return extracted_strings

def hightlight_resume(source_file, target_file, openai_api_key):
    openai_output = get_resume_hightlight(source_file, openai_api_key)
    texts = extract_hightlight_sentence(openai_output)
    hightlight_content(texts, source_file, target_file)

def hightlight_sop(source_file, target_file, openai_api_key):
    openai_output = get_sop_hightlight(source_file, openai_api_key)
    texts = extract_hightlight_sentence(openai_output)
    hightlight_content(texts, source_file, target_file)

    
api_key = "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"

#source_file = "./Materials/Resume.pdf"
#target_file = "./Materials/Resume_hightlighted.pdf"
#hightlight_resume(source_file, target_file, api_key)

source_file = "./Materials/SOP.pdf"
target_file = "./Materials/SOP_hightlighted.pdf"
hightlight_sop(source_file, target_file, api_key)

#texts = ['It was in Pattern Recognition, one of my university courses, that I witnessed the awesome performance of supervised and unsupervised learning models.']
#hightlight_content(texts, source_file, target_file)