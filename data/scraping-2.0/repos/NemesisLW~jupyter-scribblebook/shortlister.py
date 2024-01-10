# Start Evaluation of candidates and shortlist them Button
import PyPDF2
import tkinter as tk
from tkinter import filedialog

import json
import ast

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain




def rank_and_shortlist(description, llm):

    root = tk.Tk()
    root.withdraw()

    pdf_paths = filedialog.askopenfilenames(title="Select PDF files", filetypes=[("PDF files", "*.pdf")])

    extracted_text = ""

    for pdf_path in pdf_paths:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extract_text() + '\n'

        pdf_file.close()

    print(extracted_text)




    prompt_template_name=PromptTemplate(
        
        input_variables=['CV_text'],
        template="""The Following is a series of a cv texts in a continuous manner of different candidates:
    {CV_text}

    Your task is to extract the key details, such as [name, email, skills, experience and projects] for each candidate. Remember the text has details of multiple candidate.

    You should follow the following the rules while extracting details:
    1. Do not include schooling details in experiences.
    2. Try to summarise the details whenever you can.

    Your Response should follow the following format:

    [{{"name": "Name of first candidate", "email": "email of first candidate", "skills": "skills of first candidate", "experiences of first candidate", "projects": "Summary of project details of the candidate","extras":"If there is any other relevant information"}},]

    Make sure you do not add any false information. If you do not find the required values for the keys, Say "Not applicable." 

    Remember, you must not output any other text other than the JSON object.
    """
        
    )
    prompt_template_name.format(CV_text=extracted_text)


    chain=LLMChain(llm=llm,prompt=prompt_template_name)
    response=chain.run(extracted_text)


    candidate_dict = json.loads(response)


    shortlisttemplate= """As an Expert Resume Screener, you have excellent skills to screen and shortlist candidates based on the skills, experience,  previous projects and education. You are provided with a job description and details of the candidates as a JSON object. 

    Now your task is to:

    1. Rank the CVs according to their alignment with the job requirements and shortlist candidates. 
    2. Provide additional information in short bullet points on each shortlisted candidate separately.

    The following is an example of how you should format your response as JSON object:
    Example:
    {{"CV_Ranking": ["Name of Candidate No.3", "Name of Candidate No.5", "Name of Candidate No.2"],
    "Additional_Information": [
            {{"name": "Name of Candidate No.3","email": "email of Candidate No.3", "info": ["Additional Information 1 about skills", "Additional Information 2 about experiences", "Additional Information 3 about Projects"],}},
            {{"name": "Name of Candidate No.5","email": "email of Candidate No.5", "info": ["Additional Information 1 about skills", "Additional Information 2 about experiences", "Additional Information 3 about Projects", "any other additional information"],}},
            {{"name": "Name of Candidate No.2","email": "email of Candidate No.2", "info": ["Additional Information 1 about skills", "Additional Information 2 about experiences", "Additional Information 3 about Projects"],}}]}}
            
    This is the end of Example.

    Now, here is the Job Description: 
    {description}

    The following the details of the candidates who applied: 
    {candidates}

    procced to execute your task. Do not generate any extra text other than the JSONs."""

    shortlist_template = PromptTemplate(input_variables=["description", "candidates"], template=shortlisttemplate)
    shortlistChain = LLMChain(llm=llm, prompt=shortlist_template, output_key="shortlist")

    shortlist_chain = SequentialChain(chains=[shortlistChain], input_variables=["description", "candidates"], output_variables=["shortlist"], verbose=True)
    shortlist = shortlist_chain({"description": description, "candidates": candidate_dict })

    short=shortlist['shortlist']

    short = ast.literal_eval(short)


    return short