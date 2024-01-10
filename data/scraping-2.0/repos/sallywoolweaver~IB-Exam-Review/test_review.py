import pdfplumber
import openai
import re
from dotenv import load_dotenv
import os

# Load API keys from .env file
load_dotenv()

openai.api_key =  os.getenv("OPEN_API_KEY")

def evaluate_answer(prompt, student_answer, correct_answer, marks):
    chat_prompt = [
        {"role": "system", "content": "You are a helpful assistant that can evaluate the correctness of student answers. Provide detailed feedback and grade based on the markschemes. Tell the student how many marks they would have received, and if they did not get full marks how they could improve their answer."},
        {"role": "user", "content": f"Question: {prompt} (Total marks: {marks})\nStudent Answer: {student_answer}\nCorrect Answer: {correct_answer}\nProvide marks based on the markscheme and provide detailed feedback."},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_prompt,
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.5,
    )
    result = response['choices'][0]['message']['content']

    return result

def extract_answers(pdf_path):
    answers = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            temp_answer = ""
            for line in lines:
                if line.strip() and line[0].isdigit() and line[-1] == ']':
                    if temp_answer:
                        answers.append(temp_answer)
                    temp_answer = line
                elif line.strip() and line[0] == '(' and line[-1] == ']':
                    if temp_answer:
                        answers.append(temp_answer)
                    temp_answer = line
            if temp_answer:
                answers.append(temp_answer)

    return answers

def extract_marks(line):
    marks_match = re.search(r'\[(\d+)\]', line)
    return int(marks_match.group(1)) if marks_match else 0


def extract_questions(pdf_path):
    questions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            temp_question = ""
            for line in lines:
                if line.strip() and line[0].isdigit() and line[-1] == ']':
                    if temp_question:
                        marks = extract_marks(temp_question)
                        questions.append((temp_question, marks))
                    temp_question = line
                elif line.strip() and line[0] == '(' and line[-1] == ']':
                    if temp_question:
                        marks = extract_marks(temp_question)
                        questions.append((temp_question, marks))
                    temp_question = line
            if temp_question:
                marks = extract_marks(temp_question)
                questions.append((temp_question, marks))

    return questions

test_file = "G:\My Drive\Comp Sci\My Programs\ib exam python\IB-Exam-Review\Computer_science_paper_1__SL_Nov_2019.pdf"
markscheme_file = "G:\My Drive\Comp Sci\My Programs\ib exam python\IB-Exam-Review\Computer_science_paper_1__SL_markscheme_Nov_2019.pdf"

questions = extract_questions(test_file)
answers = extract_answers(markscheme_file)

for i, (question, marks) in enumerate(questions):
    print(f"Question : {question}")
    mark_scheme = answers[i]
    #print(f"Mark scheme: {mark_scheme}")
    # Get the student's answer
    student_answer = input("Enter your answer (type 'skip' to skip): ")

    if student_answer == 'skip':
        print("Skipped.")
    else:
        # Evaluate the answer and provide feedback
        feedback = evaluate_answer(question, student_answer, mark_scheme, marks)
        print(feedback)

