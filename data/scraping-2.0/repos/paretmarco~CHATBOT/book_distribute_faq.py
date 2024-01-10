import openai
import json
import time
from config import OPENAI_API_KEY
from google_sheets_helper import create_google_sheet, read_data_from_sheet, write_data_to_sheet, find_sheet_by_title

openai.api_key = OPENAI_API_KEY
MODEL = "gpt-3.5-turbo"

with open('config.json') as config_file:
    config = json.load(config_file)
    author_personality = config['author_personality']
    language = config['language']
    sheet_name = config['sheet_name']  # Read the sheet_name from the config file

def load_questions():
    try:
        with open("drafts/best_questions.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("best_questions.txt not found.")
        return []

def get_most_relevant_questions(chapter_title, chapter_content, questions, num_questions=2):
    formatted_questions = ' / '.join(questions)
    
    prompt = [
        {"role": "system", "content": "You are an AI language model and can help determine which questions are the most relevant to a given book chapter."},
        {"role": "user", "content": f"The chapter is titled '{chapter_title}' and its content is '{chapter_content}'. Given this information, which {num_questions} of the following questions are the most relevant? : {formatted_questions}"}
    ]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=prompt,
        temperature=0.7,
        max_tokens=100,  # Increased max_tokens to 100
        stop=None,
    )
    return response.choices[0].message.content.strip()

questions = load_questions()
used_questions = set()  # Keep track of used questions

# Find the Google Sheet with the specified name
folder_id = "1O4__jxbTubCObzCCn08oPe0bc0DlP-uO"  # Replace with the folder ID of the "AI_CREATED_BOOKS" directory
sheet_id = find_sheet_by_title(sheet_name, folder_id)

# Define the sheet ID and the range for reading data
READ_RANGE = "A2:C"  # Adjust this range based on your sheet's structure
sheet_data = read_data_from_sheet(sheet_id, READ_RANGE)

for index, record in enumerate(sheet_data):
    title = record[1]  # Read the chapter title from column B (index 1)
    chapter_content = record[2]  # Read the chapter content from column C (index 2)

    available_questions = [q for q in questions if q not in used_questions]  # Remove used questions
    relevant_questions = get_most_relevant_questions(title, chapter_content, available_questions)
    print(f"Added relevant questions for '{title}': {relevant_questions}")

    # Update the used_questions set
    for question in relevant_questions.split('\n'):
        used_questions.add(question.strip())

    # Write the relevant questions to column H (index 7) in the Google Sheet
    write_data_to_sheet(sheet_id, f"H{index + 2}", [[relevant_questions]])

print("Done adding relevant questions to each chapter.")