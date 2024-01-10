import os
import pandas as pd
import textract
import openai
import io
import PyPDF2

def set_openai_api_key(api_key: str):
    openai.api_key = (api_key)

def read_pdf(file_path: str):
    """
    Read a PDF file and extract text using GPT-3.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    
    doc = textract.process(file_path)
    text = doc.decode('utf-8')
    return text

def process_pdf(pdf_file):
        pdf_text = ""
        with io.BytesIO(pdf_file) as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()
            return pdf_text

def query(text: str, question: str,model = "gpt-3.5-turbo") -> str:
    """
    Query the given text with a question using GPT-2 and return the result.

    Args:
        text (str): The text to query against.
        question (str): The question to query with.

    Returns:
        str: The result of the query.
    """
    messages = [
    {"role": "system", "content": "You will be initially given text from a pdf file followed by a question related to that text. You must answer correctly based on the proper context"},]
    messages.extend([{"role": "user", "content": text} ])
    messages.extend([{"role": "user", "content": question} ])
    # Generate an answer using GPT-3
    answer = openai.ChatCompletion.create(model=model, messages=messages)

    return answer['choices'][0]["message"]["content"]

