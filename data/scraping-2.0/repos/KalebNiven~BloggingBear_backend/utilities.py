import openai
import os

from flask import jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Authenticate using the credentials file
credentials_file = "json/job-scraping-key.json"
credentials = service_account.Credentials.from_service_account_file(credentials_file,
                                                                    scopes=['https://www.googleapis.com/auth/drive',
                                                                            'https://www.googleapis.com/auth/documents'])
# Create a Google Drive API service
global drive_service
drive_service = build('drive', 'v3', credentials=credentials)
global docs_service
docs_service = build('docs', 'v1', credentials=credentials)


def generate_instruction(row_data, headline, keyword, first_run, gpt_version):
    # Use the data from 'row_data' to construct your instruction template
    style = row_data['Style']  # Column 'C'
    title = row_data['Blog Title']  # Column 'A'
    additional_data = ""
    if row_data['Facts'] is not None:
        additional_data = row_data['Facts']  # Column 'N'

    # Constructing the core instruction
    instruction = (
        f"You are an amazing writer, probably one of the best in the US."
        f"You are super versatile and can write basically on any subject and pick up any style.\n\n"
        f"Today, you are writing in the following style: '{style}'\n\n"
        f"So, keeping in mind this style, I want you to write a few sections for a blog titled: '{title}'\n\n"
        f"You’ll have to write a few sections for that post now.  Keep titles the same even though they are boring, "
        f"if the title is just one word, keep it as one word, don't add anything else. Make titles title case,"
        f"don't add any other formatting, asterisk or hashtag signs, any of that. "
        f" Please also ensure there’s no mumbling - go straight to the point. For instance, if the title is a question, "
        f"the writing should clearly respond to this question (sort of like a perfect SEO snippet will do).  Also, "
        f"please don’t write any intros and outros unless it is specifically listed as one of the headline. "
        f" Here are the headlines to write about: '{headline}'\n\n"
        f"Please use the following keywords (no need to make it bold): '{keyword}'"
    )

    # Adding the additional data instruction for the first run
    if first_run and additional_data:
        instruction += (
            f"Please ensure to integrate the facts and figures I've provided as hyperlinks directly in the text. The "
            f"anchor text should always be a number (priority)  or a fact (choose 1-3 words maximum), leading the "
            f"readers to the source without breaking the flow of the narrative. No footnotes, please. Use the "
            f"following facts/links: “{additional_data}'"
        )
    instruction_data = {
        'instruction': instruction,
        'GPT_Version': gpt_version,
    }
    return instruction_data


def formulate_instructions(row_data, run_number):
    """
    Formulates the instructions for content generation based on the data from a specific row in the Google Sheet.
    """
    instruction = []
    for key, value in row_data.items():
        first_run = False
        if key.startswith("Headlines_"):
            headline = row_data.get(key)
            if key == "Headlines_1":
                first_run = True
            if len(headline) > 2:
                keyword = row_data.get("Keywords_" + key.split('_')[1])
                gpt_version = row_data.get("GPT_v_" + key.split('_')[1])
                instruction.append(generate_instruction(row_data, headline, keyword, first_run, gpt_version))
    return instruction


def generate_content(openai_api_key, instructions, max_tokens=550):
    """
    Generates content using the OpenAI API.
    """
    openai.api_key = openai_api_key
    print(instructions)
    try:
        if instructions.get('GPT_Version'):
            gpt_model = instructions.get('GPT_Version')
        else:
            gpt_model = "gpt-3.5-turbo"

        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system",
                 "content": 'You are a kick-ass writer who can write in absolutely any style and about any subject. And you are about to write a blog post.'},
                {"role": "user", "content": instructions.get('instruction')}
            ]
        )
        # max_tokens = max_tokens
        return response
    except Exception as e:
        return e


def create_google_doc(document_title, folder_id):
    file_metadata = {
        'name': document_title,
        'parents': [folder_id],
        'mimeType': 'application/vnd.google-apps.document'
    }
    new_document = drive_service.files().create(body=file_metadata).execute()
    return new_document["id"], new_document["name"]


def update_google_doc(doc_id, content):
    requests = [
        {
            'insertText': {
                'location': {
                    'index': 1
                },
                'text': content
            }
        }
    ]
    docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
