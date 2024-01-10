import os
import re
import datetime
import argparse
import openai 
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from config import OPENAI_API_KEY
from promptTemplates import (summarizerSystemPrompt, questionGeneratorSystemPrompt, 
                             questionCritiquerSystemPrompt, convertToMongoDBSystemPrompt)

# Configuration setup
openai.api_key = OPENAI_API_KEY

def extract_all_text_in_data_directory(directory="data/"):
    """Extracts all text from PDF and TXT files in the specified directory."""
    all_text = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        
        # Extract text from PDFs
        if file_name.endswith('.pdf'):
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    all_text.append(page.extract_text())
            print(f"Successfully processed {file_name}")
        
        # Extract text from text files
        elif file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as text_file:
                all_text.append(text_file.read())
    return ''.join(all_text)

def openai_request(system_prompt, context):
    """Helper function to handle OpenAI API calls."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
    )
    return response['choices'][0]['message']['content']

def generate_summary(text):
    """Generates a summary for the given text."""
    prompt = f"This is the input below:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": summarizerSystemPrompt},
            {"role": "user", "content": prompt},
        ]
    )
    return response['choices'][0]['message']['content']

def generate_questions(text):
    """Generates questions for the given text."""
    prompt = f"Make questions on the following content:\n{text}"
    return openai_request(questionGeneratorSystemPrompt, prompt)

def critique_questions(text):
    """Generates critiques for the given set of questions."""
    prompt = f"Critique the set of questions generated:\n{text}"
    return openai_request(questionCritiquerSystemPrompt, prompt)

def finalize_questions(text, critiques):
    """Revises the questions based on provided critiques."""
    prompt = f"This is the content you're making questions on: \n{text}\nThese are the critiques you've received: \n{critiques}. Your revised questions are:"
    return openai_request(questionGeneratorSystemPrompt, prompt)

def convert_to_mongoDB(text):
    """Converts the questions to a MongoDB format."""
    prompt = f"Convert the questions to MongoDB format:\n{text}"
    return openai_request(convertToMongoDBSystemPrompt, prompt)

def text_to_pdf(text, pdf_filename="questions.pdf"):
    """Converts a text to a PDF."""

    # Split the text into individual questions
    questions = re.split(r'\d+\.', text)[1:]  # Split by numbers followed by a dot
    
    # Create a new PDF document
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    
    # Define styles for the PDF
    styles = getSampleStyleSheet()
    
    # Create an empty list to hold the PDF content
    content = []
    
    for question in questions:
        lines = question.strip().split('\n')
        for line in lines:
            content.append(Paragraph(line.strip(), styles['Normal']))
            content.append(Spacer(1, 12))  # Add a space after each line for clarity
        content.append(PageBreak())
    
    # Build the PDF document with the content
    doc.build(content)

def study_guide_to_pdf(text, pdf_filename="study_guide.pdf"):
    """Converts a study guide to a PDF."""

    styles = getSampleStyleSheet()

    # Custom styles for different parts of the document
    title_style = styles["Heading1"]
    subtitle_style = styles["Heading2"]
    normal_style = styles["Normal"]
    bullet_style = ParagraphStyle(
        "bullet",
        parent=styles["BodyText"],
        spaceBefore=0,
        leftIndent=20,
        spaceAfter=0,
    )

    content = []
    
    # Split sections by blank lines
    sections = re.split(r'\n\n', text)
    
    for section in sections:
        # If it's a structured outline or bullet point details
        if re.match(r'[IVX]+\.', section) or "- " in section:
            items = section.split('\n')
            for item in items:
                # Check if it's a main point (e.g., I., II., etc.)
                if re.match(r'[IVX]+\.', item):
                    content.append(Paragraph(item, subtitle_style))
                else:
                    # It's a subpoint or bullet point
                    content.append(Paragraph(item, bullet_style))
            content.append(Spacer(1, 12))
        else:
            # General paragraphs or titles
            lines = section.strip().split('\n')
            for line in lines:
                # If it's a title (like "Executive Summary:" or "Key Insights Extraction:")
                if line.endswith(":"):
                    content.append(Paragraph(line, title_style))
                else:
                    content.append(Paragraph(line, normal_style))
                content.append(Spacer(1, 12))

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    doc.build(content)

def create_new_folder():
    """Creates a new folder in the 'output' directory and returns its path."""
    output_dir = "output"
    new_folder_name = str(len(os.listdir(output_dir)))
    new_folder_path = os.path.join(output_dir, new_folder_name)
    os.mkdir(new_folder_path)
    return new_folder_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process data directory to generate summary, questions, etc.")
    parser.add_argument("-d", "--data_directory", help="path to data directory", default="data/")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    input_text = extract_all_text_in_data_directory(args.data_directory)
    if args.verbose:
        print("Extracted all text from the data directory.")

    summary = generate_summary(input_text)
    print("The summary is completed.")
    if args.verbose:
        print(summary)

    questions = generate_questions(summary)
    print("The questions are generated.")
    if args.verbose:
        print(questions)

    critiques = critique_questions(questions)
    print("The questions are critiqued.")
    if args.verbose:
        print(critiques)

    finalized_questions = finalize_questions(summary, critiques)
    print("The questions are finalized.")
    if args.verbose:
        print(finalized_questions)

    mongoDB_format = convert_to_mongoDB(finalized_questions)
    print("The questions are converted to JSON.")
    if args.verbose:
        print(mongoDB_format)

    output_path = create_new_folder()
    date_str = datetime.datetime.now().strftime("%m_%d")

    text_to_pdf(finalized_questions, os.path.join(output_path, f"questions_{date_str}.pdf"))
    study_guide_to_pdf(summary, os.path.join(output_path, f"study_guide_{date_str}.pdf"))
    if args.verbose:
        print(f"Saved PDFs to {output_path}")

    # Save the MongoDB file to a text file
    with open(os.path.join(output_path, f"mongoDB_{date_str}.json"), 'w') as f:
        f.write(mongoDB_format)
    if args.verbose:
        print(f"Saved MongoDB format to {output_path}/mongoDB_{date_str}.json")

