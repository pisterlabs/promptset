import openai
import json
import time
import re
from config import OPENAI_API_KEY
from google_sheets_helper import read_data_from_sheet, find_sheet_by_title, write_data_to_sheet


with open('config.json') as config_file:
    config = json.load(config_file)
    author_personality = config['author_personality']
    sheet_name = config['sheet_name']  # Read the sheet_name from the config file
    language = config['language']

def load_draft_summary():
    try:
        with open("drafts/draft_summary.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("draft_summary.txt not found.")
        return ""

draft_summary = load_draft_summary()

# Find the Google Sheet with the specified name
folder_id = "1O4__jxbTubCObzCCn08oPe0bc0DlP-uO"  # Replace with the folder ID of the "AI_CREATED_BOOKS" directory
sheet_id = find_sheet_by_title(sheet_name, folder_id)

# Define the sheet ID and the range for reading data
READ_RANGE = "A2:D"  # Adjust this range based on your sheet's structure
sheet_data = read_data_from_sheet(sheet_id, READ_RANGE)

openai.api_key = OPENAI_API_KEY
MODEL = "gpt-4"

def generate_slogan(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=prompt,
                temperature=0.7,
                max_tokens=600,
                stop=None,
            )
            slogan = response.choices[0].message.content.strip()
            return slogan
        except Exception as e:
            print(f"Error during API call: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this slogan.")
                return "Error: Unable to generate slogan."

def generate_completion(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=prompt,
                temperature=0.7,
                max_tokens=300,
                stop=None,
            )
            completion = response.choices[0].message.content.strip()
            return completion
        except Exception as e:
            print(f"Error during API call: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this completion.")
                return "Error: Unable to generate completion."

from google_sheets_helper import write_data_to_sheet

output_file_path = "C:\\D\\documenti\\AI\\program24\\book_created\\combined_output.txt"
with open(output_file_path, "w", encoding='utf-8') as output_file:
    for index, record in enumerate(sheet_data):
        chapter_title = record[1]  # Assuming the title is in column B (index 1)
        chapter_content = record[2]  # Assuming the chapter is in column C (index 2)
        chatbot_response = record[3]  # Assuming the chatbot response is in column D (index 3)

        slogan_prompt = [
            {"role": "system", "content": f"{author_personality}. You speak {language}. You are creative and generate catchy slogans or citations to introduce the book chapters."},
            {"role": "user", "content": f"Use the language {language}. Generate a professional and intelligent slogan or citation for {chapter_title} in the same language as {draft_summary}."}
        ]
        slogan = generate_slogan(slogan_prompt)

        chatbot_response_prompt = [
            {"role": "system", "content": "{author_personality}. You are a writer and a person expert in the discipline that provides practical points and suggestions for the book chapters."},
            {"role": "user", "content": f"Conclude this text '''{chapter_content} {chatbot_response}''' writing in the language: {language} the last part of this text with a concluding metaphor, and a summary in bullet points of other 50 words and telling after the main points and what a person can get from it without adding concepts but just explaining what has already been written ." }
        ]
        chatbot_response_completion = generate_completion(chatbot_response_prompt)

        if re.match(r'^\d+\.\d+', chapter_title):
            output_file.write(f"## {chapter_title} ##\n")
        else:
            output_file.write(f"# {chapter_title} #\n")
        output_file.write(f"# {slogan}\n")
        output_file.write("\n")
        output_file.write(f"{chapter_content}\n")
        output_file.write("# \n")
        output_file.write(f"{chatbot_response}\n")
        output_file.write(f"{chatbot_response_completion}\n")
        output_file.write("\n\n")

        # Flush the file buffer to write the content immediately
        output_file.flush()

        # Update Google Sheets data
        write_data_to_sheet(sheet_id, f"E{index + 2}", [[slogan]])
        write_data_to_sheet(sheet_id, f"F{index + 2}", [[chatbot_response_completion]])

        print(f"Content generated for {chapter_title} and written to combined_output.txt.")

print("Updated Google Sheets data saved.")

