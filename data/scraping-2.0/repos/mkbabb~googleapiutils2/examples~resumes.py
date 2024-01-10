"""
Provides automation to handle resume processing using Google APIs and OpenAI models. 
The functions are designed to work with Google Drive and Sheets, performing the following tasks:

- Locates all resumes in a specified Google Drive folder.
- Downloads each resume, if in PDF format, and extracts text from it.
- Utilizes OpenAI's GPT-3.5-turbo model to parse the text and obtain the candidate's name.
- Updates or adds candidate details, such as name and resume link, in a Google Sheet.
- Downloads and stores the resumes locally.
"""

import os
import tempfile
from pathlib import Path
from typing import *

import openai
import pandas as pd
import pypdf

from googleapiutils2 import Drive, Sheets

# Set the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize Google Drive and Sheets interfaces
drive = Drive()
sheets = Sheets()


def chat_completion(prompt: str, system_msg: str, model: str = "gpt-3.5-turbo") -> str:
    # Obtain completion from OpenAI based on the prompt and system message
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    # Extract the content if available
    try:
        content = response.choices[0].message.content
        return content if content is not None else ""
    except:
        return ""


def parse_name(file) -> str:
    # Check if the file is a PDF; return an empty string if not
    if file["mimeType"] != "application/pdf":
        return ""

    # System message to instruct the OpenAI model for extracting name
    system_msg = """Parse the following resume and extract only the name of the candidate.
        Format the name as follows: "First Last" (e.g. "John Doe").
        """

    # Download the file and extract the text from PDF
    with tempfile.NamedTemporaryFile() as temp:
        drive.download(file["id"], temp.name)

        pdf = pypdf.PdfReader(temp.name)
        text = "\n".join(page.extract_text() for page in pdf.pages)
        text = text[:256]

        return chat_completion(prompt=text, system_msg=system_msg)


# Links to the resumes folder and candidates sheet
resumes_folder = (
    "https://drive.google.com/drive/u/1/folders/1LlqDKpjpALra1KXuo2IMSm72g3-YhTAZ"
)
candidates_sheet = "https://docs.google.com/spreadsheets/u/1/d/1yCYmezEngDMm6z7pSkXxg7355K0Zl4_ANoUA8ghxKqY/edit?usp=drive_web&ouid=115445544450678250296"

# Get candidate details from Sheets
candidates_df = sheets.to_frame(sheets.values(candidates_sheet, "Sheet1"))
if candidates_df is None:
    raise Exception("No candidates found")

# Process each file in the resumes folder
for file in drive.list(resumes_folder):
    filepath = Path(file["name"])
    email = filepath.stem
    link = file["webViewLink"]

    candidate = candidates_df[(candidates_df["email"] == email)]

    # Update candidates details if found; else add a new entry
    if not candidate.empty:
        candidates_df.loc[candidate.index, "resume link"] = link

        if candidate["name"].iloc[0] == "":
            candidates_df.loc[candidate.index, "name"] = parse_name(file)
    else:
        name = parse_name(file)
        candidates_df = pd.concat(
            [
                candidates_df,
                pd.DataFrame(
                    {
                        "name": [name],
                        "email": [email],
                        "resume link": [link],
                    }
                ),
            ],
            ignore_index=True,
        )

# Update the Sheet with the modified candidate details
sheets.update(candidates_sheet, "Sheet1", sheets.from_frame(candidates_df))

# Download and store the resumes locally
# out_folder = Path("./resumes")
# for file in drive.list(resumes_folder):
#     out_filepath = out_folder / file["name"]
#     drive.download(
#         file["id"],
#         out_filepath,
#     )
