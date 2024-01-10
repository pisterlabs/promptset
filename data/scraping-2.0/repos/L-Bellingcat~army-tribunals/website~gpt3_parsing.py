import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from pdfminer.high_level import extract_text

from embed import trim_text

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_structured_data(text: str, max_tokens=1000):
    prompt = "Extract the names of the civilian and military judge, their names are in the format: 'Hon’ble <name>'." \
             "Remove from their names any titles and honorifics, such as Admiral, or Justice." \
             "Determine the date, format DDMMYYYY." \
             "Determine the name of the applicant, removing military titles and role." \
             "Return the data in the form of a python dictionary " \
             "{'civilian judge': <name>, 'military judge': <name>, 'date': <date>, 'applicant name': <applicant name>}."

    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are legal text bot, you are given a legal document and you must extract the required information from it."
            },
            {"role": "user", "content": trim_text(text, max_tokens=max_tokens)},
            {"role": "user", "content": prompt},
        ],
        presence_penalty=-1,
    )

    return output["choices"][0]["message"]["content"]


def summarise(text: str, max_tokens=1000):
    prompt = "Summarise the judgement in 4 sentences."

    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are legal text bot, you are given a legal document and you must summarise it in 4 sentences."
            },
            {"role": "user", "content": trim_text(text, max_tokens=max_tokens)},
            {"role": "user", "content": prompt},
        ],
        presence_penalty=-1,
    )

    return output["choices"][0]["message"]["content"]


def main():
    load_dotenv()

    # Select a random PDF file from inside the judgement_pdfs directory
    # pdf_path = random.choice(list(Path('judgement_pdfs').glob('**/*.pdf')))
    pdf_path = Path('../judgement_pdfs/kolkata/Court-58.pdf')

    # # Extract text from the PDF
    # text = extract_text(pdf_path)[:1000]
    #
    # # Count words in text
    # words = [len(line.split()) for line in text.splitlines()]
    #
    # # Print the number of words
    # print(f"Words in PDF: {sum(words)}")
    text = extract_text(pdf_path)
    print(summarise(text))


if __name__ == '__main__':
    main()
