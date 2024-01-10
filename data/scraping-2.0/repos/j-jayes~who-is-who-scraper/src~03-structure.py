# This file structures the data after it has been scraped and processed

import os
from openai import OpenAI
import json
from pathlib import Path
import dotenv
import concurrent.futures

# Get the current working directory (root directory of the project)
# root_dir = Path.cwd()
# data_dir = root_dir / "data"
data_dir = "./data"
dotenv.load_dotenv()

# Check if script is running within GitHub Actions
if os.environ.get("GITHUB_ACTIONS") == "true":
    openai_api_key = os.environ.get("OPEN_AI_KEY")
    if not openai_api_key:
        raise ValueError("No OpenAI API key found in environment variables!")
else:
    # Access the value of OPEN_AI_KEY from the environment variables
    openai_api_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=openai_api_key)

schema = {
    "type": "object",
    "required": [
        "full_name",
        "location",
        "occupation",
        "birth_details",
        "education",
        "career",
        "family",
    ],
    "properties": {
        "full_name": {"type": "string"},
        "location": {"type": "string"},
        "occupation": {"type": "string"},
        "birth_details": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "place": {"type": "string"},
                "parents": {"type": "string"},
            },
            "required": ["date", "place"],
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "year": {"type": "string"},
                    "institution": {"type": "string"},
                },
                "required": ["degree", "year", "institution"],
            },
        },
        "career": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "string"},
                    "years": {
                        "type": "object",
                        "properties": {
                            "start_year": {"type": "integer"},
                            "end_year": {"type": "integer"},
                        },
                        "required": ["start_year", "end_year"],
                    },
                    "organization": {"type": "string"},
                },
                "required": ["position", "years"],
            },
        },
        "family": {
            "type": "object",
            "properties": {
                "spouse": {"type": "string"},
                "marriage_year": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "birth_year": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["spouse", "marriage_year", "children"],
        },
        "publications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "year": {"type": "string"},
                    "type": {"type": "string"},
                },
            },
        },
        "community_involvement": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "organization": {"type": "string"},
                    "years": {"type": "string"},
                },
            },
        },
        "board_memberships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "string"},
                    "organization": {"type": "string"},
                    "years": {"type": "string"},
                },
            },
        },
        "honorary_titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "institution": {"type": "string"},
                    "year": {"type": "string"},
                },
            },
        },
        "hobbies": {"type": "array", "items": {"type": "string"}},
    },
}


def structure_biography_info(page_text):
    try:
        # Translate the Swedish text to English
        structure_prompt = f"Task: read the schema and return RFC compliant JSON information about the Swedish individuals from the 1950 biographical dictionary 'Vem är Vem' that is provided below. Use a numeric index for each biography in your JSON output and return information about all of them, including all career information available. Keep the biographic descriptions in Swedish and remove any abbreviations based on your knowledge, e.g. 'fil. kand.' is 'filosofie kandidat', and 'Skarab. l.' is 'Skaraborgs Län'. Put years in full based on context. Put dates in dd/mm/yyyy format where possible. If there is no information for a key, leave it out. If there is no information for a required key, put NULL as the value.\nHere is the schema: {schema}.\nHere is the text: {page_text}. Go!"
        structure_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert on Swedish biographies.",
                },
                {"role": "user", "content": f"{structure_prompt}"},
            ],
        )

        structured_biography_info = json.loads(
            structure_response.choices[0].message.content
        )

        return structured_biography_info

    except Exception as e:
        print(f"Error in structure_biography_info: {e}")
        return None


def process_file(input_file_path, output_file_path):
    if output_file_path.exists():
        print(f"Output file already exists, skipping: {output_file_path}")
        return

    if not input_file_path.exists():
        print(f"Input file not found: {input_file_path}")
        return

    print(f"Processing file: {input_file_path}", flush=True)

    with open(input_file_path, "r", encoding="utf-8") as file:
        page_text_in = file.read()

    structured_biography_info = structure_biography_info(page_text_in)

    if page_text_in is not None and structured_biography_info is not None:
        data = {
            "original": page_text_in,
            "structured": structured_biography_info,
        }

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=4)

        print(f"Processed file: {input_file_path}", flush=True)
    else:
        print(f"Error processing file {input_file_path}: structuring failed. Check the API response for more information.")

def process_json_files(data_dir, file_range):
    books = ["gota48", "gota65", "norr68", "skane48", "skane66", "sthlm45", "sthlm62", "svea64"]
    letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    start_index, end_index = file_range

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for book in books:
            for letter in letters:
                for file_index in range(start_index, end_index + 1):
                    input_file_path = Path(data_dir) / f"joined_text/{book}/{letter}/{book}_{letter}_{file_index}.txt"
                    output_file_path = Path(data_dir) / f"json_structured/{book}/{letter}/{book}_{letter}_structured_{file_index}.json"
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    futures.append(executor.submit(process_file, input_file_path, output_file_path))

        for future in concurrent.futures.as_completed(futures):
            future.result()  # This is just to catch exceptions if any occurred during the file processing


# File range for testing
FILE_RANGE = [120, 140]

# Call the function with the appropriate data directory and file range
process_json_files(data_dir, FILE_RANGE)