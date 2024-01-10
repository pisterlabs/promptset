from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from io import BytesIO
import requests
import openai
import os
from dotenv import load_dotenv
from fuzzywuzzy import fuzz 

load_dotenv()

app = FastAPI()

# Initialize a dictionary to store previous questions and their answers
previous_questions = {}

# Pydantic model for the response data
class TextResponse(BaseModel):
    text: str

class QuestionRequest(BaseModel):
    question: str
    pdf_content: str
    #extracted_text: str 

# Function to extract text from a PDF using PyPDF2
def extract_text_pypdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def ngrok_nougat(pdf_url,ngrok_url):
    try:
        # Download the PDF file from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()

        # Create a file-like object from the response content
        file_data = response.content

        # Prepare the file for uploading
        files = {'file': ('uploaded_file.pdf', file_data, 'application/pdf')}

        # Replace with the ngrok URL provided by ngrok
        ng_url = ngrok_url 

        # Send the POST request to the Nougat API via ngrok
        response = requests.post(f'{ng_url}/predict/', files=files, timeout=300)

        # Check if the request to the Nougat API was successful (status code 200)
        if response.status_code == 200:
            # Get the response content (Markdown text)
            markdown_text = response.text
            return markdown_text
        else:
            return f"Failed to make the request! Status Code: {response.status_code}"

    except Exception as e:
        return f"An error occurred: {e}"

@app.post("/extract-pypdf-text")
async def extract_pypdf_text(pdf_link:str):
    try:
        
        # Download the PDF file
        response = requests.get(pdf_link)
        pdf_content = response.content

        # Extract text using PyPDF
        text = extract_text_pypdf(BytesIO(pdf_content))
        return JSONResponse(content={"text": text})
    except Exception as e:
        print("Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/extract-nougat-text")
async def extract_nougat_text(pdf_link:str, ngrok_url:str):
    try:
        # Download the PDF file
        response = requests.get(pdf_link)
        pdf_content = response.content

        # Extract text using PyPDF
        result = ngrok_nougat(pdf_link, ngrok_url)
        return JSONResponse(content={"text": result})
    except Exception as e:
        print("Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    try:
        # Get the question from the request
        question = request.question
        pdf_content = request.pdf_content

        # Check if the same or similar question has been asked before
        best_match_question = None
        best_match_score = 0

        for prev_question in previous_questions:
            similarity_score = fuzz.ratio(question, prev_question)
            if similarity_score > best_match_score:
                best_match_score = similarity_score
                best_match_question = prev_question

        if best_match_question and best_match_score > 80:
            # If a similar question has been asked, return the answer from the similar question
            answer = previous_questions[best_match_question]
        else:
            # If it's a new question, send the question to the OpenAI API
            openai.api_key = os.getenv("OPEN_AI_KEY")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Assist me with answers only if the question is relevant to provided content, if not reply me with \"Sorry, your question is out of context\"."
                    },
                    {
                        "role": "user",
                        "content": pdf_content
                    },
                    {
                        "role": "assistant",
                        "content": question
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            # Extract and return the answer
            answer = response['choices'][-1]['message']['content']

            # Store the new question and its answer in the dictionary
            previous_questions[question] = answer

        return JSONResponse(content={"answer": answer})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    