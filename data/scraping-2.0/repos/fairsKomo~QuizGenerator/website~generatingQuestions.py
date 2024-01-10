from PyPDF2 import PdfReader
import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

TEMPLATE = """
            questions: [
            {
                "id": 1,
                "question": "How many people in the earth?"
                "options": [
                    7.888 billions,
                    2 billions,
                    3 billions,
                    15 billions
                ],
                "correct_answer": 7.888 billions
            },
            {
                "id": 2,
                "question": "the king of football in 2012 was Lionel Messi?"
                "options": [
                    True,
                    False
                ],
                "correct_answer": True
            }
            ]
"""

def loadPdf(pdfFile):
    reader = PdfReader(pdfFile)
    myText = ""

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        content = page.extract_text()
        if content:
            myText+=content
    
    return myText


def get_response(text,num = 5):
    prompt = f"""
        Make yourself proffesional in creating question based on
        text that is delimitted with four hashtags, I need you to
        generate {num} MCQ questions.
        Your response must be in json format. Each question contains
        id, question, 4 options as a list and the correct answer.
        This is template of how it should be formatted: {TEMPLATE}
        
        the text is: ````{text}````
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role":"system",
                "content":prompt,
            }
        ],
    )
    return json.loads(response["choices"][0]["message"]["content"])

def get_response_tf(text,num = 5):
    prompt = f"""
        Make yourself proffesional in creating question based on
        text that is delimitted with four hashtags, I need you to
        generate {num} True or False questions.
        Your response must be in json format. Each question contains
        id, question, 2 options True or False as a list and the correct answer.
        This is template of how it should be formatted: {TEMPLATE}
        
        the text is: ````{text}````
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role":"system",
                "content":prompt,
            }
        ],
    )
    return json.loads(response["choices"][0]["message"]["content"])

def get_response_mix(text,num = 5):
    prompt = f"""
        Make yourself proffesional in creating question based on
        text that is delimitted with four hashtags, I need you to
        generate {num} MCQ and True or False questions.
        Your response must be in json format. Each question contains
        id, question, 4 options if it is MCQ or 2 options if it is True or False as a list and the correct answer.
        Please make sure that half of the questions will be MCQ and the Other half is True or False, forexample
        if there`s 4 questions 2 will be MCQ and 2 will be True or False
        This is template of how it should be formatted: {TEMPLATE}
        
        the text is: ````{text}````
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role":"system",
                "content":prompt,
            }
        ],
    )
    return json.loads(response["choices"][0]["message"]["content"])

