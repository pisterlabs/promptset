import openai
import json


def generate_screening_question(job_title, job_description, resume_text):
    messages = [
        {"role": "system", "content": """You are a helpful assistant. Your purpose is to make 10 Screening Questions for the candidate appearing for the interviews. The screening questions should be based on the job description and candidate's resume. The questions can be based on situations, technical capabilities and soft skills.  The format of the screening questions should be as follows - 
                1. 3 Project and previous employment Based Questions from CV. Questions can be based on situations or how they solved various issues faced
                2. 3 Questions based on the technical skills of the candidate. These questions should be in depth inside the concept and try to ask relevant detailed question. These questions will test the technical capability of the candidate and how much candidate knows about the topic in detail. Try to test the concept of the candidate in depth. For example if it is SQL, then you ask question related to primary and foreign keys and their working. Also, do not give reference to CV while asking these questions.
                3. 3 Questions based on job profile. These questions can be of any type but the theme should be strictly around job profile provided
                4. 1 Question on Soft Skills
                
                Input will be in the following format - 
                Job Title - Job Title for the position will be provided here
                Job Description - job description will be provided here
                Resume - Candidate's resume will be provided here
                 
                Output should be in the form of json object with the following keys - 
                Question - The screening question to be asked should be here
                Reference - the reference of the question
                Type - Job Profile, Technical, Example & Project based or Soft Skill
                Tag - List of tags associated with the question for categorizing it and using it for future interviews
                Return the output in the json format separated by comma. For example -
                [
                    {
                        "Question": "What is your name?",
                        "Reference": "Resume",
                        "Type": "Example & Project based",
                        "Tag": ["Name", "Example & Project based"]
                    },
                    ...
                ]
                """},
        {"role": "user", "content": f"""Job Title - {job_title}, Job Description - {job_description}, Resume - {resume_text}"""}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response['choices'][0]['message']['content']
