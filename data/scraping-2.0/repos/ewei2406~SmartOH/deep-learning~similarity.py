import os

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from util import validate_email, send_email_with_pdf, get_google_drive_service, upload_file_to_drive
from transcribe import AI
import glob
from generate_pdf import PDFGenerator

file_paths = glob.glob('./data/**/*.m4a', recursive=True)

# Load environment variables
load_dotenv()

# Initialize SentenceTransformer model and OpenAI API key
model = SentenceTransformer('all-MiniLM-L6-v2')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

def get_help_question(question):

    messages = [ {"role": "system", "content": 
                "You are a teaching assistant for the course."} ]

    prompts = [
        f"A student is asking you for help with the following question: {question}. What are 3 steps one can take to effectively help the student and explain the concepts to them without immediately giving away the answer?",
        "Provie 3 sentences that would help the student most following the steps."
    ]
    for i in range(2):
        message = prompts[i]

        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        if i == 1:
            return reply
        messages.append({"role": "assistant", "content": reply})

    return reply


class HelpInput(BaseModel):
    question: str

@app.post("/help")
def calculate_similarity(data: HelpInput):
    
    question = data.question

    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    try:
        reply = get_help_question(question)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calc_similarity_function(target_string: str, string_list: list[str]):
    # Compute embeddings for both lists
    embeddings1 = model.encode([target_string], convert_to_tensor=True)
    embeddings2 = model.encode(string_list, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.tolist()

def compute_current_topic(question_list):

    messages = [ {"role": "system", "content": 
                "You are a intelligent assistant."} ]

    prompts = [
        f"You are a bot designed to summarize a list of questions from students in the waiting list for a computer science office hours. Given the below questions, what are three words or less that would be most helpful for other students to know to identify whether they have similar problems? The questions are:  {str(question_list)}",
        "How would you reduce all of these problems to the THREE questions (NO MORE) that would most comprehensively cover them, separated by commas?  If homework or administrative matters are mentioned in a question, they must be included."
    ]
    for i in range(2):
        message = prompts[i]

        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        if i == 1:
            return reply
        messages.append({"role": "assistant", "content": reply})

    return reply

class SimilarityInput(BaseModel):
    target_string: str
    string_list: list[str]
"""
    {
        "target_string": "What is Dijkstra's algorithm?",
        "string_list": [
            "How does bubble sort work?",
            "What is dynamic programming?",
            "How do you find the shortest path in a weighted graph?"
        ]
    }
"""
@app.post("/similarity")
def calculate_similarity(data: SimilarityInput):
    
    target_string = data.target_string
    string_list = data.string_list

    if not target_string or not string_list:
        raise HTTPException(status_code=400, detail="Both target_string and string_list must be provided")

    try:
        similarity_scores = calc_similarity_function(target_string, string_list)
        return {"similarity_scores": similarity_scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class TopicInput(BaseModel):
    questions: list[str]

"""
    {
        "questions": [
            "How does bubble sort work?",
            "What is dynamic programming?",
            "How do you find the shortest path in a weighted graph?"
        ]
    }
"""
@app.post("/current-topic")
def get_current_topic(data: TopicInput):

    question_list = data.questions

    if not question_list:
        raise HTTPException(status_code=400, detail="No questions provided")

    try:
        current_topic = compute_current_topic(question_list)
        return {"current_topic": current_topic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class EmailInput(BaseModel):
    email: str

"""
    {
        "student_email": "dqx2gm@virginia.edu",
    }
"""
@app.get("/send-report")
def send_report(data: EmailInput):

    student_email = data.email

    if not validate_email(student_email):
        raise HTTPException(status_code=400, detail="Invalid email address")

    try:
        """
        generating PDF
        """

        # Initialize the AI object
        for file_path in file_paths:
            ai = AI(file_path, './data/sa_speech.json')
            transcription_result = ai.transcribe()
            print(f'Transcription for {file_path}: {transcription_result}')
            pdf_generator = PDFGenerator(transcription_result)
            pdf_generator.run_conversation()

            # pdf is created in /deep-learning/temp.pdf
            print("PDF generated")
            # upload to google drive
            service = get_google_drive_service()
            upload_file_to_drive(service, './temp.pdf', 'application/pdf')
        
            # send_email_with_pdf(student_email, "./data/report.pdf")
        return {"message": "Report sent successfully to " + student_email}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# def generate_report(transcription):
#     messages = [ {"role": "system", "content": 
#                 "You are a intelligent assistant."} ]

#     prompts = [
#         "Your job is to take the transcript of a conversation, then summarize it and format it into a brief latex document capturing the questions and the core of the answers.  YOU MUST INCLUDE BOTH THE QUESTION AND THE ANSWER FOR EACH QUESTION THAT IS ASKED OR TOPIC THAT IS DISCUSSED. Here is the transcript:"
#     ]
#     for i in range(1):
#         message = prompts[i]

#         if message:
#             messages.append(
#                 {"role": "user", "content": message},
#             )
#             chat = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo", messages=messages
#             )
#         reply = chat.choices[0].message.content
#         print(f"ChatGPT: {reply}")
#         if i == 1:
#             return reply
#         messages.append({"role": "assistant", "content": reply})

#     return reply