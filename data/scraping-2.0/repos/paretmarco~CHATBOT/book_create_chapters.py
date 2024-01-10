import openai
import json
import csv
import os
import time
import regex
from config import OPENAI_API_KEY
import logging
from google_sheets_helper import create_google_sheet, read_data_from_sheet, write_data_to_sheet, find_sheet_by_title
from create_custom_prompt import create_custom_prompt

with open('config.json') as config_file:
    config = json.load(config_file)
    author_personality = config['author_personality']
    language = config['language']

# Use the author_personality variable in your program

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up API key and model
openai.api_key = OPENAI_API_KEY
MODEL = "gpt-4"

# Function to generate a chapter
def generate_chapter(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=prompt,
                temperature=0.7,
                max_tokens=1350,
                stop="\n\n",
            )
            content = response['choices'][0]['message']['content'].strip()
            
            # Log the user prompt and the received answer
            logging.info(f"User prompt sent: {prompt[-1]['content']}")
            logging.info(f"Answer received: {content}")

            return content
        except Exception as e:
            logging.warning(f"Error during API call: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Skipping this chapter.")
                return "Error: Unable to generate chapter content."

# Read the draft_text.txt file
with open("drafts/draft_text.txt", "r", encoding="utf-8") as f:
    draft_text = f.read().strip()
    draft_text = regex.sub(r'\\u([0-9a-fA-F]{4})', lambda x: chr(int(x.group(1), 16)), draft_text)

# Function to generate a summary
def generate_summary(text, retries=3, delay=5):
    summary_prompt = [
        {"role": "system", "content": f"{author_personality}. You are a highly skilled summarizer and you speak {language}. Summarize the following text using the same language:"},
        {"role": "assistant", "content": text},
        {"role": "user", "content": f"Write in the following language: {language} and provide a summary of the main points in at least 150 tokens of the following ###{draft_text}### you must tell in the summary clearly what this text want to convey and demostrate and why it is important."},
    ]

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=summary_prompt,
                temperature=0.7,
                max_tokens=250,
                stop=None,
            )
            summary = response['choices'][0]['message']['content'].strip()
            
            # Log the user prompt and the received answer
            logging.info(f"User prompt sent: {summary_prompt[-1]['content']}")
            logging.info(f"Summary received: {summary}")

            return summary
        except Exception as e:
            logging.warning(f"Error during API call: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Skipping summary generation.")
                return "Error: Unable to generate summary."

# Generate a summary of the draft_text
draft_summary = generate_summary(draft_text)

# Save the generated summary to a file
with open("drafts/draft_summary.txt", "w", encoding="utf-8") as f:
    f.write(draft_summary)
    logging.info(f"{draft_summary}")

# Read the input_queries.txt file and remove empty lines
with open("input/input_queries.txt", "r") as f:
    lines = [regex.sub(r'\\u([0-9a-fA-F]{4})', lambda x: chr(int(x.group(1), 16)), line.strip()) for line in f if line.strip()]
    logging.info(f"Read {len(lines)} lines from input_queries.txt")

# Initialize the JSON and CSV data structures
json_data = []
csv_data = [["number", "title", "chapter"]]

# Create the output directory if it doesn't exist
output_directory = "book_created"
os.makedirs(output_directory, exist_ok=True)

def update_json_data(number, title, chapter):
    if os.path.exists(json_filename):
        with open(json_filename, "r") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = []

    json_data.append({"number": number, "title": title, "chapter": chapter})

    with open(json_filename, "w") as json_file:
        json.dump(json_data, json_file)

# Remove the book.txt file if it exists and create a new one
book_txt_path = os.path.join(output_directory, "book.txt")
if os.path.exists(book_txt_path):
    os.remove(book_txt_path)

# Remove the book_data.json file if it exists and create a new one
json_filename = os.path.join(output_directory, "book_data.json")
if os.path.exists(json_filename):
    os.remove(json_filename)

# Create a Google Sheet with the specified name if it doesn't exist
# Read the book_title.txt file
with open("drafts/book_title.txt", "r", encoding="utf-8") as f:
    book_title = f.read().strip()

# Create the sheet_name with the number of lines and the content of the book_title.txt file
sheet_name = f"Book-{len(lines)} {book_title}"
folder_id = "1O4__jxbTubCObzCCn08oPe0bc0DlP-uO"  # Replace with the folder ID of the "AI_CREATED_BOOKS" directory
sheet_id = find_sheet_by_title(sheet_name, folder_id)
if not sheet_id:
    sheet_id = create_google_sheet(sheet_name, folder_id)

# Save the sheet_name to the config.json file
config['sheet_name'] = sheet_name
with open('config.json', 'w') as config_file:
    json.dump(config, config_file, ensure_ascii=False)

# Define the sheet ID and the range for writing data
WRITE_RANGE = "A2:C"  # Adjust this range based on your sheet's structure

with open(json_filename, "w") as json_file:
    json.dump([], json_file)

with open(book_txt_path, "a") as book_file:

    # Iterate through the lines and generate chapters
    for index, line in enumerate(lines):
        line = line.strip()

        previous_chapter = draft_text
        current_chapter = line
        custom_prompt_text = create_custom_prompt(previous_chapter, line, current_chapter)

        prompt = [
            {"role": "system", "content": f"{author_personality}. You are an absolute expert in this discipline and you prepare a book. You speak the language with ease. You write sentences full of meanings and crafted with a lot of images."},
            {"role": "assistant", "content": draft_text},
            {"role": "user", "content": f"{author_personality}. You are writing a book with this index {lines}. Continue developing in a clear way the specific concept for this specific chapter {line} in this language: {language}. You must grab immediately the attention and make the content unique for {line} differentiating it from the former and following chapters. You are the author and you prefer the first person connecting the concepts if possible to events of your life and giving both evocative examples as from everyday life as with images in the Oscar Wilde style. Please write a detailed text of at least 55 words, simple to understand even for a 15-year-old and in the same language as ###{draft_summary}###. Integrate these instructions {custom_prompt_text} as well. Write in the same language as  ###{draft_summary}###"},
                ]

        chapter = generate_chapter(prompt)

        # Write the chapter to the comprehensive book file
        book_file.write(f"# {line}\n")
        book_file.write(chapter)
        book_file.write("\n\n")
        print(f"Appended chapter '{line}' to '{os.path.join(output_directory, 'book.txt')}'.")  # Added logging

        # Append the chapter to the JSON and CSV data structures
        json_data = {"number": index + 1, "title": line, "chapter": chapter}
        csv_data.append([index + 1, line, chapter])

        logging.info(f'Generated chapter: {line}')

        # Update the JSON data with the new chapter
        update_json_data(index + 1, line, chapter)

        # Update the Google Sheet with the new chapter
        write_data_to_sheet(sheet_id, f"A{index + 2}:C{index + 2}", [[index + 1, line, chapter]])

# Write the CSV file
with open(os.path.join(output_directory, "book_data.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

# Write the chapter data to the Google Sheet
print(json_data)
formatted_data = [[record["number"], record["title"], record["chapter"]] for record in json_data]
write_data_to_sheet(sheet_id, WRITE_RANGE, formatted_data)

logging.info(f"Writing chapter data to Google Sheet with ID: {sheet_id} and range: {WRITE_RANGE}")
