import os, sys
local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
from pathlib import Path
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)

config = load_config(Path(local_python_path) / "config.json", add_date=False)
from docx import Document
from docx2python import docx2python
import openai
openai.api_key = config['openai_api_key']

def extract_endnotes(doc_path):
    """
    Extracts the endnotes from a Word document and returns them as a list.
    """
    with docx2python(doc_path) as docx_content:
        endnotes = docx_content.endnotes
    return endnotes

def convert_to_chicago_style(note):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"I am going to give you a footnote in Hebrew. Please convert to Hebrew Chicago style. Remember, keep the note in Hebrew, and return the result in utf-8 string\n\nthe note is:\n{note}\n",
      max_tokens=150  # Adjust as needed
    )
    return response.choices[0].text.strip()


def main():
    docs = Path(config['input_dir']).glob("*.docx")
    for doc in docs:
        logger.info(f"Processing {doc}")
        notes = extract_endnotes(doc)
        insert_footnotes_and_bibliography(doc, notes)

if __name__ == '__main__':
    doc_path = input("Enter the path to the Word document: ")
    notes = extract_endnotes(doc_path)
    